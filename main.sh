#!/usr/bin/env bash

set -eu

OUTPUT_DIR="outputs/child_dataset"
TRACE_DIR="${OUTPUT_DIR}/trace"

if ! python scripts/bootstrap_assets.py --check-only >/dev/null 2>&1; then
  echo "Offline assets are missing."
  echo "Run the bootstrap step first:"
  echo "  export HF_TOKEN=your_token"
  echo "  python scripts/bootstrap_assets.py"
  exit 1
fi

# Fixed strongest path:
# Demucs htdemucs_ft -> ClearerVoice MossFormer2_SE_48K -> pyannote turns
python scripts/build_dataset.py \
  --input-dir data/audio \
  --output-dir "${OUTPUT_DIR}" \
  --seed 20260409 \
  --num-threads 8 \
  --min-turn-sec 0.35 \
  --turn-merge-gap-sec 0.35 \
  --turn-glitch-max-sec 0.25 \
  --turn-glitch-gap-sec 0.2 \
  --child-threshold 0.6 \
  --max-gap-seconds 30 \
  --multi-link-threshold 0.7 \
  --max-turns 6 \
  --trace-dir "${TRACE_DIR}"