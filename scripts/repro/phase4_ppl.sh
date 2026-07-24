#!/usr/bin/env bash
# Phase 4: wikitext-2 PPL 비교 (LaRoSA 재현 수치와 동일 프로토콜, seqlen 4096).
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

model=meta-llama/Llama-2-7b-hf
cfg=config/llama-2-7b-hf_default.json

echo "=== [ppl] FULL ==="
python -u utils/ppl_wikitext.py --model_name $model --method full --config_file $cfg

echo "=== [ppl] R_SPARSE searched s=0.5 ==="
python -u utils/ppl_wikitext.py --model_name $model --method r_sparse \
    --sparse_config_file config/llama2_sparsity_50_evolutionary_search.npy --config_file $cfg

echo "=== [ppl] R_SPARSE uniform s=0.5 rho=0.7 ==="
python -u utils/ppl_wikitext.py --model_name $model --method r_sparse \
    --target_sparsity 0.5 --sparse_ratio 0.7 --config_file $cfg

echo "=== [ppl] R_SPARSE uniform s=0.4 rho=0.7 (LaRoSA@40% 비교용) ==="
python -u utils/ppl_wikitext.py --model_name $model --method r_sparse \
    --target_sparsity 0.4 --sparse_ratio 0.7 --config_file $cfg

echo "=== [ppl] done ==="
