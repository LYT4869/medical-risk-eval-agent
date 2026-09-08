#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_ID="intfloat/multilingual-e5-small"
readonly MODEL_REVISION="614241f622f53c4eeff9890bdc4f31cfecc418b3"
readonly CACHE_DIR="${TREESEM_HF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
readonly EXPORT_BACKEND="${TREESEM_ROUTING_EXPORT_BACKEND:-onnx_int8}"
readonly OUTPUT_ROOT="${TREESEM_ROUTING_OUTPUT_ROOT:-artifacts/agent-routing}"
readonly EXPORT_IMAGE="${TREESEM_ROUTING_EXPORT_IMAGE:-treesem-routing-export:local}"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$EXPORT_BACKEND" != "onnx_fp32" && "$EXPORT_BACKEND" != "onnx_int8" ]]; then
  echo "TREESEM_ROUTING_EXPORT_BACKEND must be onnx_fp32 or onnx_int8" >&2
  exit 2
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "routing source model cache is missing: $CACHE_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
readonly OUTPUT_ABSOLUTE="$(cd "$OUTPUT_ROOT" && pwd)"

docker build \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  -f "$PROJECT_ROOT/deploy/docker/routing-export.Dockerfile" \
  -t "$EXPORT_IMAGE" "$PROJECT_ROOT"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$CACHE_DIR:/models/huggingface:ro" \
  -v "$OUTPUT_ABSOLUTE:/artifacts" \
  "$EXPORT_IMAGE" \
  --backend "$EXPORT_BACKEND" \
  --model "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --tasks /app/config/tasks.yaml \
  --parity-cases /app/evaluation/routing_cases.json \
  --thresholds /app/config/routing_thresholds.json \
  --output-root /artifacts
