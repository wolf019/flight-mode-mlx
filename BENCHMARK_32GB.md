# BENCHMARK_32GB.md — 32 GB tier validation, measured

> **What shipped vs. what we projected.**
>
> The 128 GB rig's projection for the 32 GB tier was: **MoE Qwen3.6-35B-A3B-4bit
> at 16k context with kv=4, peak ~22.4 GB, decode ~95 t/s.** Validated on real
> 32 GB hardware (Apple M5, 32 GB, macOS 26.3.1):
>
> - **Memory math holds within 1%.** Peak 22.18 GB at 16k kv=4 (vs 22.37 GB
>   projected, −0.8%). 32k peak 23.38 GB (vs 23.52 GB projected, −0.6%). No
>   allocator misbehavior under pressure.
> - **Decode throughput is roughly half the projection** (43 t/s at 16k kv=4
>   vs 95 t/s on the M4 Max). Cause: the M5 in this MacBook Pro has **10 GPU
>   cores vs the M4 Max's 40** — 4× fewer cores, ~½ the throughput. This is a
>   GPU-tier effect, not a 32 GB memory-pressure effect.
> - **The 32 GB-tier claim survives intact.** Every measured cell is above
>   the 25 t/s feasibility threshold, including a 48k-context cell that the
>   128 GB-rig recommendation didn't even consider feasible.
> - **Validated, shippable recommendation for 32 GB:** MoE 32k context, kv=4,
>   peak **23.38 GB**, decode **31.8 t/s**. More aggressive than the 128 GB
>   rig's conservative "16k comfortable" call; still leaves >8 GB headroom.
>
> The "local AI is more accessible than the discourse suggests, including on
> 32 GB MacBooks" claim is defensible. The honest qualifier the post should
> add: decode rate scales with GPU core count, so the smaller-tier laptops
> in the lineup will run at half the M4 Max's tokens-per-second — still well
> above what readers will perceive as "real-time."

## 1. Hardware attestation

| Field | Value |
|---|---|
| Machine | MacBook Pro, Mac17,2 |
| Chip | Apple **M5** (10-core GPU) |
| Unified memory | 32 GB (34,359,738,368 bytes) |
| macOS | 26.3.1 (build 25D2128) |
| Kernel | Darwin 25.3.0 (xnu-12377.91.3~2 / RELEASE_ARM64_T8142) |
| Power source | AC (verified `pmset -g batt` before each run) |
| Python | 3.12.12 (uv-managed: `~/.local/share/uv/tools/mlx-lm/bin/python3`) |
| mlx | 0.31.2 |
| mlx-metal | 0.31.2 |
| **mlx-lm** | **0.31.3** (matches the 128 GB rig exactly) |
| numpy | 2.4.4 |
| transformers | 5.6.2 |
| fastapi / uvicorn / sse-starlette | 0.136.1 / 0.46.0 / 3.3.4 |

### Divergence from the 128 GB rig

| | 128 GB rig (BENCHMARK_FULL.md baseline) | This 32 GB machine |
|---|---|---|
| Chip | M4 Max (40-core GPU) | **M5 (10-core GPU)** |
| RAM | 128 GB | 32 GB |
| macOS | 25.5.0 | **26.3.1** |
| `mlx` | 0.31.3 | **0.31.2** (one patch behind) |
| `mlx-lm` | 0.31.3 | 0.31.3 (match) |

The M5 microarchitecture is one generation _newer_ than the M4 Max but ships
with 4× fewer GPU cores in this MacBook Pro tier. That's the dominant
explanation for the throughput gap below; memory measurements are unaffected
by GPU core count and line up with the projection.

## 2. Task 1 — recommended config: MoE 16k context, both kv settings

`scripts/bench.py` matrix at `--contexts 16384 --kv-configs none 4 --gen-tokens 32`,
all other flags unchanged from the 128 GB harness (`prefill_step=2048`,
`kv_group_size=64`, `quantized_kv_start=0`).

Raw JSON: `benchmark-results/bench_32gb_task1_16k.json`.

### Measured vs. 128 GB projection

| ctx | kv | metric | 128 GB projection | M5 32 GB measured | Δ |
|---:|---|---|---:|---:|---:|
| 16,384 | none | peak MLX GB | 22.56 | **22.42** | **−0.6 %** ✓ |
| 16,384 | none | prefill t/s | 1556.2 | 1001.1 | −35.7 % ⚠ |
| 16,384 | none | decode t/s | 108.4 | 50.6 | **−53.3 %** ⚠ |
| 16,384 | 4    | peak MLX GB | 22.37 | **22.18** | **−0.8 %** ✓ |
| 16,384 | 4    | prefill t/s | 1358.8 | 913.3 | −32.8 % ⚠ |
| 16,384 | 4    | decode t/s | 95.0 | 43.0 | **−54.7 %** ⚠ |

Peak RSS sat at 1.16 GB across both runs (the bench script measures the
Python process's RSS, not unified memory; MLX peak is the right number).

### Hypothesis for the throughput delta

Throughput on Apple silicon is GPU-core-bound for batch-1 decode of large
4-bit-weight models. The M4 Max has 40 GPU cores; the M5 in this Mac17,2
configuration has 10. A 4× gap in GPU cores explaining ~½ throughput
(rather than ¼) means the M5 generation's per-core uplift is real — it's
recovering ~2× per-core relative to the M4 — but it can't fully close the
gap. Memory is independent of core count, which is why the peak figures
match within noise.

The brief asked us to flag any divergence > 10 %; both throughput cells
exceed that threshold and are flagged here. The conclusion is **not** that
the 32 GB machine is memory-pressured — peaks match the projection — but
that the 128 GB rig's _decode_ headlines were specifically the M4 Max
ceiling, not a generic "32 GB tier" ceiling. Future hardware-tier
recommendations should pair the RAM tier with a GPU-core-count caveat.

## 3. Task 2 — ceiling probe (24k → 32k → 48k, kv=4)

Started at 24k and walked context up. No OOM, no thermal throttling
observed, no degradation between consecutive cells.

Raw JSON: `benchmark-results/bench_32gb_task2_ceiling.json`.

| ctx | peak MLX GB | prefill t/s | decode t/s | feasible (≥25 t/s)? |
|---:|---:|---:|---:|---|
| 24,576 | 22.78 | 800.6 | 36.3 | ✓ |
| 32,768 | **23.38** | 684.0 | **31.8** | **✓** |
| 49,152 | 24.59 | 483.3 | **25.3** | ✓ (just) |

### Findings

1. **32k peak (23.38 GB) almost exactly matches the 128 GB projection
   (23.52 GB).** Memory math is hardware-tier-independent — the
   prediction in BENCHMARK_FULL.md §6 holds.
2. **48k fits at 24.59 GB peak — 7.4 GB working-set headroom on this
   32 GB machine, even with browser/IDE realistically idle.** That's
   _more aggressive_ than what BENCHMARK_FULL.md §6 promised for the
   32 GB tier (which capped at "32k requires 6 GB headroom mode").
3. **The decode-throughput floor for "still feasible" sits between 48k
   and 64k.** 48k decodes at 25.3 t/s (right on the threshold). Pushing
   further would mean falling below the 25 t/s feasibility bar.

### Updated 32 GB recommendation

Where BENCHMARK_FULL.md §4 said:
> 32 GB tier: MoE 16k, kv=4 → peak 22.37 GB, decode 95.0 t/s

The validated, shippable claim for actual 32 GB hardware is:
> **32 GB tier: MoE 32k, kv=4 → peak 23.38 GB, decode 31.8 t/s.**
> 48k still feasible at 25.3 t/s if the demo wants the bigger context.

Decode rate is lower than the 128 GB rig predicted (GPU-core effect, not
memory) but the larger context that fits is the more interesting story.

## 4. Task 3 — server smoke test on validated config

Launched `python3 server.py --model mlx-community/Qwen3.6-35B-A3B-4bit
--kv-bits 4 --kv-group-size 64 --quantized-kv-start 0 --prefill-step 2048
--warmup-tokens 16 --port 8080`. Server logged:

```
Model loaded in 3.96s (RSS 11.31 GB, MLX peak 19.51 GB)
EOS token ids: [248044, 248046]
Warmup done in 1.33s — prefill 6.6 tok/s, decode 62.4 tok/s, peak RSS 10.52 GB
```

### `/health` response

```json
{
    "status": "ok",
    "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "kv_bits": 4,
    "kv_group_size": 64,
    "quantized_kv_start": 0,
    "max_kv_size": null,
    "prefill_step": 2048,
    "rss_gb": 10.51,
    "mlx_peak_gb": 19.59
}
```

`kv_bits: 4` confirmed — the production path is running symmetric 4-bit KV
quantization, same as the bench harness.

### Streaming chat completion

A streaming SSE request to `/v1/chat/completions` with prompt
`"Say exactly: flight mode engaged"` produced, in order:

```
data: {... "delta": {"role": "assistant"} ...}
data: {... "delta": {"content": "flight"} ...}
data: {... "delta": {"content": " mode"} ...}
data: {... "delta": {"content": " engaged"} ...}
data: {... "finish_reason": "stop" ...}
data: [DONE]
```

The streaming path works.

### `x_mlx_metrics` from a non-streaming request

Non-streaming completion with prompt `"In one sentence, what is unified
memory?"` returned a coherent answer plus this metrics block:

```json
"x_mlx_metrics": {
    "prompt_tokens": 21,
    "completion_tokens": 34,
    "prefill_secs": 0.2237,
    "decode_secs": 0.6041,
    "prefill_tps": 93.90,
    "decode_tps": 56.29,
    "peak_rss_gb": 10.53,
    "peak_mlx_gb": 19.63
}
```

### Long-context request via `scripts/demo_long_context.py`

Drove a 130k-char (~22.5k token) prompt through the running server
exactly as the demo arc would. Server-reported metrics:

| metric | value |
|---|---:|
| prompt_tokens | 22,518 |
| completion_tokens | 32 |
| prefill_secs | 27.45 |
| decode_secs | 0.86 |
| **prefill_tps** | **820.27** |
| **decode_tps** | **37.38** |
| peak_rss_gb | 10.54 |
| **peak_mlx_gb** | **22.61** |

This sits exactly between the bench-harness 16k cell (22.18 GB / 43 t/s)
and the 24k cell (22.78 GB / 36.3 t/s) — server path and bench path
agree. The model produced a coherent answer about KV-cache quantization
and unified memory as the tightest constraint, confirming generation
quality holds under long context, not just throughput.

## 5. Methodology

- Same harness as `BENCHMARK_FULL.md`: `scripts/bench.py`, greedy decode 32
  new tokens per cell, `prefill_step=2048`, `kv_group_size=64`,
  `quantized_kv_start=0`, `mx.reset_peak_memory()` between runs, NumPy
  RandomState(42) for the synthetic prompt (same prompt across kv settings).
- AC power throughout (62 % → 68 % battery state during runs, all charging).
- `caffeinate -dimsu` running in a background tab.
- `bench.py`'s built-in warmup runs once before the matrix so the measured
  cells reflect steady-state Metal kernels.
- Server launched via the uv-managed Python (the `run_server.sh` shebang
  hardcodes the 128 GB rig's `tomaxberg` path; for 32 GB validation we
  invoked `python3 server.py …` directly through the local uv tool venv).

## 6. Raw artifacts

| File | What it captures |
|---|---|
| `benchmark-results/bench_32gb_task1_16k.json` | Task 1: MoE 16k both kv |
| `benchmark-results/bench_32gb_task2_ceiling.json` | Task 2: 24k/32k/48k kv=4 |
