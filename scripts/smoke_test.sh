#!/usr/bin/env bash

# Cheap end-to-end check before a full training or evaluation job.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

EVAL_LIMIT=8 bash scripts/evaluate_models.sh

MAX_SAMPLES=16 \
EVAL_SAMPLES=8 \
TARGET_BATCH_SIZE=8 \
COLQWEN2_OUTPUT="${OUTPUT_ROOT}/colqwen2-smoke" \
bash scripts/train_colqwen2.sh

echo "Inference/evaluation and one minimal training run completed."
