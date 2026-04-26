# flight-mode local inference

Strategic demo: run Qwen3 family models locally on a MacBook Pro M4 Max
with MLX + symmetric 4-bit KV cache quantization, **with the throughput
to make long-context actually usable**.

Phase 1 measured the dense Qwen3.6-27B (apples-to-apples with what
Julien Chaumond ran on llama.cpp). Phase 2 measured the MoE
Qwen3.6-35B-A3B and found it **dominates dense on every axis** the
demo cares about: 3–6× the prefill throughput, 3–4× the decode
throughput, less peak memory at every context, and the same PPL parity
under 4-bit KV quant.

## What this is not

- Not a new model. Not a new quantization algorithm. Not a framework.
- Not flash attention; MLX already uses it internally.
- Not a mock benchmark. All numbers in `BENCHMARK.md` (Phase 1, dense)
  and `BENCHMARK_FULL.md` (Phase 2, dense + MoE + hardware tiers) are
  measured on the actual machine with seeds fixed. Raw JSON in
  `benchmark-results/`.

## What this is

- A production-shaped FastAPI server (~370 LOC, `server.py`) exposing
  OpenAI-compatible `/v1/chat/completions` over `mlx_lm.generate_step`.
  Per-request `x_mlx_metrics` returns prefill t/s, decode t/s, peak
  RSS, peak MLX memory.
- Symmetric 4-bit KV cache quantization applied from step 0 of every
  request. Quality validated on both models:
  - Dense Qwen3.6-27B: **PPL 3.413 vs 3.416 baseline** (within noise).
  - MoE Qwen3.6-35B-A3B: **PPL 3.659 vs 3.703 baseline** (within
    noise).
- Two scripts that underwrote the claims:
  - `scripts/ppl_eval.py` — chunked perplexity with configurable KV
    quant (unlike `mlx_lm.perplexity`, this actually exercises the
    cache).
  - `scripts/bench.py` — prefill/decode throughput + peak memory matrix
    across contexts, KV configs, and models.
- `PREFLIGHT.md`: recording checklist.
- `BENCHMARK.md`: Phase 1 measurements (dense, 2k–200k).
- `BENCHMARK_FULL.md`: Phase 2 deliverable — full sovereignty matrix
  across hardware tiers (32 GB / 64 GB / 128 GB), measured.
- `PHASE2_PLAN.md`: the contract Phase 2 was executed against, with
  architecture facts and time-budget reasoning.

## Quick start — MoE (the recommended config)

```bash
# Install the mlx-lm toolchain with server deps
uv tool install mlx-lm --with datasets --with fastapi --with uvicorn --with psutil --with sse-starlette

# Download the MoE model (~20 GB)
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit

# Run the server
MODEL_PATH=mlx-community/Qwen3.6-35B-A3B-4bit KV_BITS=4 ./run_server.sh

# Probe
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool

# Chat
curl -sS -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role":"user","content":"Explain unified memory in one sentence."}],
    "stream": true,
    "temperature": 0,
    "max_tokens": 80,
    "chat_template_args": {"enable_thinking": false}
  }'
```

The dense Qwen3.6-27B-4bit is also fully supported and was the Phase 1
test bed; same flags, just `MODEL_PATH=mlx-community/Qwen3.6-27B-4bit`.
The dense numbers are the apples-to-apples comparison against
llama.cpp; the MoE numbers are what the post recommends running.

## The narrative wedge, stated plainly

Sovereignty in local inference is not about buying more RAM. The
4-bit-weight Qwen3.6 family fits in 15–20 GB on disk. The question is
what you do with the remaining budget — workspace, KV cache,
concurrent contexts, browser, IDE. This stack:

1. Uses MLX's native 4-bit weight quantization (as-shipped).
2. Applies symmetric 4-bit quantization to the KV cache (new here vs
   `mlx_lm.server` in 0.31.3, which doesn't expose the flag).
3. Validates zero PPL regression from the KV quant on **both** dense
   and MoE.
4. Measures the hardware-tier matrix end-to-end: which (model, ctx, kv)
   combinations fit a 32 / 64 / 128 GB MacBook with usable headroom.
5. Ships the whole thing as a single-file OpenAI-compatible server you
   can run offline.

The single most defensible recommendation that came out of Phase 2:
**MoE Qwen3.6-35B-A3B-4bit at 200k context with kv=4 fits a 64 GB
MacBook with 18 GB of working-set headroom, decoding at 25.8 t/s.**
Full numbers and per-tier specifics in `BENCHMARK_FULL.md`.
