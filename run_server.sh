#!/usr/bin/env bash
# Launch the flight-mode server with the demo configuration.
# Every flag here is load-bearing; see BENCHMARK.md for the justifications.

set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-mlx-community/Qwen3.6-27B-4bit}"
export KV_BITS="${KV_BITS:-4}"
export KV_GROUP_SIZE="${KV_GROUP_SIZE:-64}"
export QUANTIZED_KV_START="${QUANTIZED_KV_START:-0}"
export PREFILL_STEP="${PREFILL_STEP:-2048}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8080}"

# Use the uv-managed mlx-lm interpreter by default so deps (fastapi, psutil,
# sse-starlette, datasets, mlx_lm) are consistent with the benchmark scripts.
PYTHON="${PYTHON:-/Users/tomaxberg/.local/share/uv/tools/mlx-lm/bin/python3}"

exec "$PYTHON" server.py \
    --model "$MODEL_PATH" \
    --kv-bits "$KV_BITS" \
    --kv-group-size "$KV_GROUP_SIZE" \
    --quantized-kv-start "$QUANTIZED_KV_START" \
    --prefill-step "$PREFILL_STEP" \
    --warmup-tokens 32 \
    --host "$HOST" \
    --port "$PORT"
