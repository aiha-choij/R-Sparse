#!/usr/bin/env bash
# Phase 2 (prep): Llama-2-7B low-rank weight 생성 + full baseline 8개 태스크 평가.
set -euo pipefail
source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate r_sparse
cd "$(dirname "$0")/../.."

model=meta-llama/Llama-2-7b-hf
# evaluation.py는 호출당 태스크 1개만 지원한다 (다중 태스크 → NotImplementedError)
TASKS="winogrande piqa sciq openbookqa hellaswag boolq arc_easy arc_challenge"

if [ ! -f ../low_rank_models/llama-2-7b/model.layers.31.mlp.down_proj.pt ]; then
    echo "=== [phase2] generating low-rank weights for llama-2-7b ==="
    python -u utils/prepare_low_rank_weight.py \
        --model_name $model \
        --output_dir ../low_rank_models/llama-2-7b
fi

for task in $TASKS; do
    echo "=== [phase2] FULL / $task ==="
    python -u evaluation.py \
        --tasks $task --num_fewshot 0 \
        --model_name $model --method full \
        --config_file config/llama-2-7b-hf_default.json
done

echo "=== [phase2] done ==="
