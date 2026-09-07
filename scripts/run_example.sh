#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python run_editvid.py \
  --manifest-json examples/manifest.json \
  --source-videos-dir examples/source_videos \
  --rows 0 \
  --max-frames 16 \
  --chunk-size 16 \
  --fallback-chunk-sizes 8,4 \
  --output-root outputs \
  --run-name setup-test
