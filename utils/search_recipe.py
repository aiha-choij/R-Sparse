"""Evolutionary search for the R-Sparse sparsification recipe (Algorithm 1, ICLR 2025).

Searches per-module sparse ratio alpha (rho in the paper) under a fixed model-level
sparsity budget, minimizing average perplexity over C4 calibration samples.
Group-wise strategy: variables are split into groups; one group is optimized at a
time while the others stay at the best-so-far values.

Output is a text file compatible with evaluation.py --sparse_config_file:
flat list of (alpha, s) pairs in model.named_modules() order.
"""
import os
import sys
import math
import time
import json
import random
import argparse
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import datasets as hf_datasets
from transformers import AutoConfig, AutoTokenizer

from models.modeling_llama import LlamaForCausalLM_R_Sparse, R_Sparse_Linear
from utils.setup import setup_config, get_wikitext2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    p.add_argument('--config_file', type=str, default='config/llama-2-7b-hf_default.json')
    p.add_argument('--target_sparsity', type=float, default=0.5)
    p.add_argument('--prefill_ratio', type=float, default=0.1)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=5)
    p.add_argument('--group_size', type=int, default=28)
    p.add_argument('--mutation_rate', type=float, default=0.5)
    p.add_argument('--crossover_rate', type=float, default=0.5)
    p.add_argument('--nsamples', type=int, default=16)
    p.add_argument('--seqlen', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='config/llama2_sparsity_50_evolutionary_search.npy')
    return p.parse_args()


def get_c4_windows(nsamples, seed, seqlen, tokenizer):
    ds = hf_datasets.load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    text = ' '.join(ds[i]['text'] for i in range(2000))
    enc = tokenizer(text, return_tensors='pt')
    random.seed(seed)
    windows = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        windows.append(enc.input_ids[:, i:i + seqlen])
    return windows


def module_dims(name, config):
    if 'self_attn' in name:
        return config.hidden_size, config.hidden_size
    if 'down_proj' in name:
        return config.intermediate_size, config.hidden_size
    return config.hidden_size, config.intermediate_size


def build_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    config = setup_config(config, args)
    model = LlamaForCausalLM_R_Sparse.from_pretrained(
        args.model_name, config=config, torch_dtype=torch.float16)

    s = args.target_sparsity
    mods = []
    for name, m in model.named_modules():
        if isinstance(m, R_Sparse_Linear):
            in_ch, out_ch = module_dims(name, config)
            m._dims = (in_ch, out_ch)
            # alpha=0일 때의 예산이 rank 상한 — 이 rank로 SVD 인자를 한 번만 로드해두고
            # 이후 후보마다 forward의 [:, :rank] 슬라이싱으로 재사용한다.
            m._rank_max = max(int((1 - s) * in_ch * out_ch / (in_ch + out_ch)), 1)
            m.rank = m._rank_max
            mods.append(m)
    model._load_low_rank_module(config)
    model = model.cuda().eval()
    return model, tokenizer, mods


def apply_recipe(model, mods, alphas, s, prefill_ratio, calib):
    for m, a in zip(mods, alphas):
        in_ch, out_ch = m._dims
        a = float(a)
        m.sparse_ratio = a
        m.target_sparsity = 1 - (1 - s) * a
        channels = max(int(in_ch * (1 - s) * a), 1)
        low_rank_budget = in_ch * out_ch * (1 - s) - channels * out_ch
        m.rank = min(max(int(low_rank_budget / (in_ch + out_ch)), 1), m._rank_max)
        m.flag_getting_threshold = True
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        with torch.no_grad():
            model(input_ids=calib)
    for m in mods:
        m.prefill_ratio = prefill_ratio
        if m.sparse_ratio >= 0.999:
            m.mode = 'sparse'
        elif m.sparse_ratio <= 0.001:
            m.mode = 'low_rank'
        else:
            m.mode = 'r_sparse'


def eval_ppl(model, windows):
    ppls = []
    with torch.no_grad():
        for w in windows:
            w = w.cuda()
            logits = model(input_ids=w).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                w[:, 1:].reshape(-1))
            ppls.append(math.exp(min(loss.item(), 20.0)))
    return sum(ppls) / len(ppls)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, tokenizer, mods = build_model(args)
    nvars = len(mods)
    print(f'modules: {nvars}, groups of {args.group_size}')

    windows = get_c4_windows(args.nsamples, args.seed, args.seqlen, tokenizer)
    calib = torch.cat(get_wikitext2(nsamples=1, seed=42, seqlen=512, tokenizer=tokenizer), dim=0).cuda()

    s = args.target_sparsity
    P, G, pm, pc = args.population, args.generations, args.mutation_rate, args.crossover_rate

    def fitness(vec):
        apply_recipe(model, mods, vec, s, args.prefill_ratio, calib)
        return eval_ppl(model, windows)

    best = rng.uniform(0.0, 1.0, size=nvars)
    best_fit = fitness(best)
    print(f'init random recipe ppl: {best_fit:.4f}')
    ones_fit = fitness(np.ones(nvars))
    print(f'all-sparse (alpha=1) ppl: {ones_fit:.4f}')
    if ones_fit < best_fit:
        best, best_fit = np.ones(nvars), ones_fit

    groups = [list(range(g, min(g + args.group_size, nvars)))
              for g in range(0, nvars, args.group_size)]

    t0 = time.time()
    for gi, idxs in enumerate(groups):
        pop = rng.uniform(0.0, 1.0, size=(P, len(idxs)))
        pop[0] = best[idxs]
        pop[1] = 1.0

        def group_fitness(ind):
            vec = best.copy()
            vec[idxs] = ind
            return fitness(vec)

        fits = np.array([group_fitness(ind) for ind in pop])
        for gen in range(G):
            children = []
            for i in range(P):
                x1, x2, x3 = rng.choice(P, size=3, replace=False)
                mutant = pop[x1] + pm * (pop[x2] - pop[x3])
                mask = rng.uniform(size=len(idxs)) > pc
                child = np.where(mask, mutant, pop[i])
                children.append(np.clip(child, 0.0, 1.0))
            children = np.array(children)
            child_fits = np.array([group_fitness(c) for c in children])
            allpop = np.vstack([pop, children])
            allfit = np.concatenate([fits, child_fits])
            order = np.argsort(allfit)[:P]
            pop, fits = allpop[order], allfit[order]
            print(f'[group {gi+1}/{len(groups)}] gen {gen+1}/{G} '
                  f'best ppl {fits[0]:.4f} elapsed {time.time()-t0:.0f}s', flush=True)
        if fits[0] < best_fit:
            best[idxs] = pop[0]
            best_fit = fits[0]
        print(f'[group {gi+1}/{len(groups)}] done, global best ppl {best_fit:.4f}', flush=True)

    out = np.zeros(nvars * 2)
    out[0::2] = best
    out[1::2] = s
    np.savetxt(args.output, out)
    print(f'saved recipe to {args.output}, final ppl {best_fit:.4f}')
    print('alpha mean/min/max:', best.mean(), best.min(), best.max())


if __name__ == '__main__':
    main()
