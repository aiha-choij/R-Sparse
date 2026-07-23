#!/usr/bin/env bash
# Phase 3: Llama-2-7B evolutionary search (Algorithm 1) 로 50% sparsity recipe 탐색.
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

# low-rank weight 준비 대기 (phase2_llama2_prep.sh가 생성)
deadline=$((SECONDS + 21600))
until [ -f ../low_rank_models/llama-2-7b/model.layers.31.mlp.down_proj.pt ]; do
    [ $SECONDS -ge $deadline ] && { echo "timeout waiting for low-rank weights" >&2; exit 1; }
    sleep 60
done

python -u utils/search_recipe.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --config_file config/llama-2-7b-hf_default.json \
    --target_sparsity 0.5 \
    --output config/llama2_sparsity_50_evolutionary_search.npy

echo "=== [phase3] search done ==="
