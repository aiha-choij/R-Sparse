"""WikiText-2 test perplexity (token-level, non-overlapping windows).

LaRoSA 등 다른 sparsification 논문들이 쓰는 표준 프로토콜과 동일하게 측정한다:
"\n\n".join(test) → seqlen 윈도우 분할 → 전체 토큰 평균 NLL의 exp.
"""
import os
import sys
import math
import argparse

import torch
import torch.nn as nn
import datasets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.setup import setup_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--method', type=str, default='full')
    p.add_argument('--target_sparsity', type=float, default=0.5)
    p.add_argument('--prefill_ratio', type=float, default=0.1)
    p.add_argument('--sparse_ratio', type=float, default=1)
    p.add_argument('--config_file', type=str, default='config/llama-2-7b-hf_default.json')
    p.add_argument('--sparse_config_file', type=str, default=None)
    p.add_argument('--seqlen', type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    _, tokenizer, model = setup_model(args)
    model = model.eval().to(args.device)

    test = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    enc = tokenizer('\n\n'.join(test['text']), return_tensors='pt')
    ids = enc.input_ids
    nwin = ids.shape[1] // args.seqlen

    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(nwin):
            w = ids[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device)
            logits = model(input_ids=w).logits
            nll += loss_fct(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                w[:, 1:].reshape(-1)).item()
            ntok += w.shape[1] - 1
            print(f'window {i+1}/{nwin} running ppl {math.exp(nll/ntok):.4f}', flush=True)

    print(f'FINAL wikitext2 ppl ({args.method}, s={args.target_sparsity}, '
          f'rho={args.sparse_ratio}, cfg={args.sparse_config_file}): {math.exp(nll/ntok):.4f}')


if __name__ == '__main__':
    main()
