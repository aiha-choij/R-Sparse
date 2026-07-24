#!/usr/bin/env bash
# 남은 재현 작업 직렬 실행: uniform r05 → uniform r03(fp16) → wikitext-2 PPL 4종.
# dispatcher의 GRACE_SEC 이후 CPU 웜업 중 이중 배차를 피하기 위해 한 잡으로 묶는다.
set -euo pipefail
HERE="$(dirname "$0")"

bash "$HERE/eval_llama2_rsparse.sh" uniform 0.5

RSPARSE_DTYPE=float16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$HERE/eval_llama2_rsparse.sh" uniform 0.3

bash "$HERE/phase4_ppl.sh"

echo "=== [phase2-remaining] all done ==="
