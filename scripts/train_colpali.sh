#!/usr/bin/env bash

# ColPali LoRA training with the same stable effective-batch policy as ColQwen2.

set -euo pipefail
# A batch of 32 is the stable, less memory-intensive ColPali default.
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-32}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

MODEL_NAME="${COLPALI_BASE:-vidore/colpaligemma-3b-pt-448-base}"
OUTPUT_DIR="${COLPALI_OUTPUT:-${OUTPUT_ROOT}/colpali-lora}"

ARGS=(
  scripts/train.py
  --model-type colpali
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

echo "Training ColPali with ${NUM_PROCESSES} process(es), effective batch approximately ${TARGET_BATCH_SIZE}."
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES}" "${ARGS[@]}"
else
  python "${ARGS[@]}"
fi
