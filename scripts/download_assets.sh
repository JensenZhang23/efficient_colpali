#!/usr/bin/env bash

# Download inference checkpoints and the public train/evaluation datasets.
# PaliGemma may require `hf auth login` and accepting Google's Gemma license.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${PROJECT_ROOT}"

command -v hf >/dev/null 2>&1 || {
  echo "Missing Hugging Face CLI. Run: pip install -U huggingface_hub"
  exit 1
}

echo "Downloading ColQwen2 inference checkpoint..."
hf download vidore/colqwen2-v1.0

echo "Downloading ColQwen2 training base..."
hf download vidore/colqwen2-base

echo "Downloading ColPali inference checkpoint..."
hf download vidore/colpali-v1.2

echo "Downloading ColPali training base..."
hf download vidore/colpaligemma-3b-pt-448-base

echo "Materializing the evaluation dataset..."
python scripts/download_data.py --dataset eval --output-dir "${DATA_ROOT}"

if [[ "${DOWNLOAD_TRAIN_DATA:-1}" == "1" ]]; then
  echo "Materializing the approximately 41 GB training dataset..."
  python scripts/download_data.py --dataset train --output-dir "${DATA_ROOT}"
else
  echo "Skipping the training dataset because DOWNLOAD_TRAIN_DATA=${DOWNLOAD_TRAIN_DATA:-0}."
fi

echo "Weights are cached under ${HF_HOME}; datasets are ready under ${DATA_ROOT}."
