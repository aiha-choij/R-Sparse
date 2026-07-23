#!/usr/bin/env bash
# Llama-2-7B R-Sparse 평가.
#   eval_llama2_rsparse.sh uniform <rho>   # uniform 50% sparsity, sparse_ratio=rho
#   eval_llama2_rsparse.sh searched <npy>  # search recipe 파일로 평가
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

model=meta-llama/Llama-2-7b-hf
TASKS=winogrande,piqa,sciq,openbookqa,hellaswag,boolq,arc_easy,arc_challenge
mode=$1

if [ "$mode" = "uniform" ]; then
    rho=$2
    echo "=== [eval] R_SPARSE uniform target_sparsity=0.5 rho=${rho} / 8 tasks ==="
    python -u evaluation.py \
        --tasks $TASKS --num_fewshot 0 \
        --model_name $model --method r_sparse \
        --target_sparsity 0.5 --sparse_ratio $rho \
        --config_file config/llama-2-7b-hf_default.json
elif [ "$mode" = "searched" ]; then
    npy=$2
    echo "=== [eval] R_SPARSE searched recipe ${npy} / 8 tasks ==="
    python -u evaluation.py \
        --tasks $TASKS --num_fewshot 0 \
        --model_name $model --method r_sparse \
        --sparse_config_file $npy \
        --config_file config/llama-2-7b-hf_default.json
else
    echo "unknown mode: $mode" >&2; exit 1
fi

echo "=== [eval] done ==="
