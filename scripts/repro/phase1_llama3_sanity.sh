#!/usr/bin/env bash
# Phase 1: Llama-3-8B sanity check.
# low-rank weight 생성(없을 때만) 후 piqa에서 full vs r_sparse(공개 search config) 비교.
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

model=meta-llama/Meta-Llama-3-8B

if [ ! -f ../low_rank_models/llama-3-8b/model.layers.31.mlp.down_proj.pt ]; then
    echo "=== [phase1] generating low-rank weights for llama-3-8b ==="
    python -u utils/prepare_low_rank_weight.py \
        --model_name $model \
        --output_dir ../low_rank_models/llama-3-8b
fi

echo "=== [phase1] FULL / piqa ==="
python -u evaluation.py \
    --tasks piqa --num_fewshot 0 \
    --model_name $model --method full

echo "=== [phase1] R_SPARSE 50% (evolutionary search config) / piqa ==="
python -u evaluation.py \
    --tasks piqa --num_fewshot 0 \
    --model_name $model --method r_sparse \
    --sparse_config_file config/llama3_sparsity_50_evolutionary_search.npy \
    --config_file config/llama-3-8b_default.json

echo "=== [phase1] done ==="
