#!/usr/bin/env bash

# Recommended training path: ColQwen2 is faster and generally stronger for retrieval.

set -euo pipefail
# ColQwen2-v1.0 used an effective batch of 256; preserve it by default.
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-256}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

MODEL_NAME="${COLQWEN2_BASE:-vidore/colqwen2-base}"
OUTPUT_DIR="${COLQWEN2_OUTPUT:-${OUTPUT_ROOT}/colqwen2-lora}"

ARGS=(
  scripts/train.py
  --model-type colqwen2
  --model-name "${MODEL_NAME}"
  --dataset "${TRAIN_DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --eval-samples "${EVAL_SAMPLES}"
  --report-to "${REPORT_TO}"
)

if [[ -n "${RUN_NAME}" ]]; then
  ARGS+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Training ColQwen2 with ${NUM_PROCESSES} process(es), effective batch approximately ${TARGET_BATCH_SIZE}."
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES}" "${ARGS[@]}"
else
  python "${ARGS[@]}"
fi
