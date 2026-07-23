#!/usr/bin/env bash
# Phase 3: Llama-2-7B evolutionary search (Algorithm 1) 로 50% sparsity recipe 탐색.
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

python -u utils/search_recipe.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --config_file config/llama-2-7b-hf_default.json \
    --target_sparsity 0.5 \
    --output config/llama2_sparsity_50_evolutionary_search.npy

echo "=== [phase3] search done ==="
