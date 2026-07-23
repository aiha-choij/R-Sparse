#!/usr/bin/env bash
# Llama-2-7B R-Sparse 평가.
#   eval_llama2_rsparse.sh uniform <rho>   # uniform 50% sparsity, sparse_ratio=rho
#   eval_llama2_rsparse.sh searched <npy>  # search recipe 파일로 평가
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

model=meta-llama/Llama-2-7b-hf
# evaluation.py는 호출당 태스크 1개만 지원한다
TASKS="winogrande piqa sciq openbookqa hellaswag boolq arc_easy arc_challenge"
mode=$1

# low-rank weight가 준비될 때까지 대기 (phase2_llama2_prep.sh가 생성).
# 없으면 _load_low_rank_module이 경고만 내고 진행해 잘못된 결과가 나온다.
wait_for() {
    local target=$1 deadline=$((SECONDS + 21600))
    until [ -f "$target" ]; do
        [ $SECONDS -ge $deadline ] && { echo "timeout waiting for $target" >&2; exit 1; }
        sleep 60
    done
}
wait_for ../low_rank_models/llama-2-7b/model.layers.31.mlp.down_proj.pt

if [ "$mode" = "uniform" ]; then
    rho=$2
    for task in $TASKS; do
        echo "=== [eval] R_SPARSE uniform target_sparsity=0.5 rho=${rho} / $task ==="
        python -u evaluation.py \
            --tasks $task --num_fewshot 0 \
            --model_name $model --method r_sparse \
            --target_sparsity 0.5 --sparse_ratio $rho \
            --config_file config/llama-2-7b-hf_default.json
    done
elif [ "$mode" = "searched" ]; then
    npy=$2
    wait_for "$npy"
    for task in $TASKS; do
        echo "=== [eval] R_SPARSE searched recipe ${npy} / $task ==="
        python -u evaluation.py \
            --tasks $task --num_fewshot 0 \
            --model_name $model --method r_sparse \
            --sparse_config_file $npy \
            --config_file config/llama-2-7b-hf_default.json
    done
else
    echo "unknown mode: $mode" >&2; exit 1
fi

echo "=== [eval] done ==="
