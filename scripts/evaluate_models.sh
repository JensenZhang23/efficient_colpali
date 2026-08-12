#!/usr/bin/env bash

# Evaluate both official checkpoints. Set *_MODEL to evaluate local LoRA outputs.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

COLQWEN2_MODEL="${COLQWEN2_MODEL:-vidore/colqwen2-v1.0}"
COLPALI_MODEL="${COLPALI_MODEL:-vidore/colpali-v1.2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
fi

python scripts/evaluate.py \
  --model-type colqwen2 \
  --model-name "${COLQWEN2_MODEL}" \
  --dataset "${EVAL_DATASET}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --output "${OUTPUT_ROOT}/colqwen2-results.json" \
  "${LIMIT_ARGS[@]}"

python scripts/evaluate.py \
  --model-type colpali \
  --model-name "${COLPALI_MODEL}" \
  --dataset "${EVAL_DATASET}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --output "${OUTPUT_ROOT}/colpali-results.json" \
  "${LIMIT_ARGS[@]}"

echo "Results saved under ${OUTPUT_ROOT}."
