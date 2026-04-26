# BENCHMARK.md — Qwen3.6-27B-4bit on MLX with aggressive KV management

All numbers below are measured on the machine the demo runs on. Raw logs
are in `benchmark-results/`. Anything surprising is flagged inline.

## Hardware + software attestation

| Field | Value |
|---|---|
| Machine | MacBook Pro, Apple M4 Max, 128 GB unified memory |
| macOS | 25.5.0 |
| Python | 3.12 |
| mlx | (installed alongside mlx-lm) |
| mlx-lm | 0.31.3 (upgraded from 0.31.1 mid-session) |
| Model | `mlx-community/Qwen3.6-27B-4bit` (commit `c000ac2c`) |
| Weights on disk | 15.1 GB (3× safetensors shards) |
| Model params | 26,896 M total (includes vision tower; only language_model is loaded) |

Why this model: parity with what Julien Chaumond ran on llama.cpp (same
Qwen3.6-27B weights, same 4-bit quantization), but served via MLX with KV
cache quantization layered on top — which is what the "same model, less
memory" narrative requires.

## Architecture fact that shapes the whole story

Qwen3.6-27B is **hybrid attention**: 64 decoder layers, `full_attention_interval = 4`.

- 16 layers use full attention (GQA, 4 KV heads, head_dim 256).
- 48 layers use **gated delta net** linear attention with a fixed-size
  recurrent state (handled by `ArraysCache`, not `KVCache`).

Consequence for the KV-quant claim: only 16 of 64 layers have a growing KV
cache to quantize. The linear layers contribute constant memory. The memory
savings from KV quantization therefore scale with context, not with layer
count — and are modest at short context because the KV cache on 16 layers
at 4 heads × 256 head_dim is already small.

This is also **why it runs well on 128 GB machines.** The architecture was
designed memory-aware.

## Baseline — no KV quantization

Single-shot perplexity on `allenai/tulu-3-sft-mixture`, seq_len 2048,
32 samples (65,536 tokens scored).

| Metric | Value |
|---|---|
| **Perplexity** | **3.416 ± 0.034** |
| Eval time | 374 s (via `mlx_lm.perplexity`) |
| Peak memory (MLX) | 18.78 GB |
| Peak RSS | 19.0 GB |

Also confirmed via a custom chunked eval (`scripts/ppl_eval.py`) that
mirrors how the cache is used at decode time: same **3.416 ± 0.034**. The
chunked path with cache produces the same answer as the cache-free forward.
This means the script is sound — it's the control for the quantized run
below.

## With symmetric 4-bit KV quantization

Same dataset, same split, same seed.

| Config | Perplexity | Delta vs baseline |
|---|---|---|
| No KV quant (baseline) | 3.416 ± 0.034 | — |
| **KV 4-bit, group 64, start 0** | **3.413 ± 0.034** | **−0.003 (within noise)** |

**No quality loss at 4-bit symmetric KV.** This is the result that
matters. The error bars overlap completely; the point estimate is
fractionally *lower* which is noise.

Why this works, briefly: Qwen3.6-27B is a 27 B model pre-quantized to
4-bit weights. At that model scale, quantization error on the KV cache is
small relative to the error already baked into the weight quant. The
TurboQuant paper found K-sensitivity dominates on smaller models
(Qwen2.5-7B) with aggressive combined quantization; here the combination
is less aggressive (mlx's affine quant, not TurboQuant's rotation + QJL)
and the model is larger, so symmetric 4-bit holds.

**Asymmetric K=8/V=4 therefore NOT implemented.** No demonstrated need.
If this were Qwen2.5-7B or a model showing PPL regression > ~2 %, the
asymmetric path would be required (and mlx_lm's uniform `QuantizedKVCache`
would be insufficient — it takes a single `bits` value).

## Throughput matrix

Random-token prompts of the listed length, greedy decode of 64 new tokens.
Kernel compilation warmed beforehand. Same process, `mx.reset_peak_memory()`
between runs so the peak-memory column is per-run.

<!-- TABLE_THROUGHPUT_START -->
| kv_bits | ctx    | prefill t/s | decode t/s | peak MLX GB | Δ peak vs no-quant |
|---------|-------:|------------:|-----------:|------------:|-------------------:|
| none    |  2,048 |       243.7 |       29.5 |       18.21 |                  — |
| none    |  8,192 |       208.7 |       27.3 |       19.22 |                  — |
| none    | 32,768 |       186.6 |       22.2 |       23.24 |                  — |
| 4       |  2,048 |       189.5 |       27.9 |       18.21 |               0.00 |
| 4       |  8,192 |       191.6 |       25.6 |       18.84 |              −0.38 |
| 4       | 32,768 |       177.6 |       20.2 |       21.71 |              **−1.53** |
| none    | 65,536 |       168.7 |       21.9 |       28.65 |                  — |
| 4       | 65,536 |       157.8 |       16.7 |       25.56 |              **−3.09** |
| none    | 200,000|       104.7 |       13.2 |       51.28 |                  — |
| 4       | 200,000|        94.8 |        7.8 |       41.38 |              **−9.90** |
<!-- TABLE_THROUGHPUT_END -->

Reading the table:

- **`peak MLX GB`** is the clean per-run peak (via `mx.reset_peak_memory()` between
  runs). It is the authoritative memory number.
- **`rss_gb` intentionally omitted** — on Apple silicon, MLX allocations go through
  the Metal allocator and do not reliably show up in Python's `psutil` RSS. Use
  `/usr/bin/time -l` wrapped around the *whole process* to cross-check, or trust
  `mx.get_peak_memory()` for per-run numbers.
- **Prefill throughput drops ~22 % with KV quant at 2 k context and recovers at
  longer contexts.** The quantize-on-write + quantized-matmul overhead is fixed per
  chunk; with longer prefills that overhead amortizes. Not "free," but the
  memory-efficiency story is unaffected and this is what to say out loud.
- **Peak memory savings scale with context**, as expected. At 2 k the KV cache on
  16 full-attention layers is under 130 MB, smaller than the run-to-run noise
  floor. At 8 k the 4-bit cache saves 0.38 GB vs unquantized. At 32 k it saves
  **1.53 GB** — matching the prediction from the architecture math below.

The arithmetic, so the numbers are inspectable. KV cache per full-attention layer
per token = 4 KV heads × 256 head_dim × 2 (K + V) × 2 bytes (bf16) = 4 KB.
Across 16 full-attn layers that is 64 KB per token. At context N in bf16:
64 KB × N bytes. At 4-bit affine with group 64: ≈ (64 / 16 + metadata overhead)
KB per token. So 32 k context: 2.0 GB bf16 → ~0.5 GB 4-bit, saving ~1.5 GB.
This matches the measured 1.53 GB delta in the kv=4 / 32 k row.

65 k measured delta: **−3.09 GB** (28.65 unquantized → 25.56 quantized). The
architecture math predicted ~3.0 GB based on KV-cache-only accounting; the
0.09 GB extra likely comes from the quantized storage having slightly smaller
allocator arenas.

200 k measured delta: **−9.90 GB** (51.28 unquantized → 41.38 quantized). The
KV-cache-only math predicts a 9.8 GB saving (16 full-attn layers × 64 KB/token
× 200 k tokens × ¾ from 4-bit), and the measurement matches it within noise.
Two notes that shaped how the 200 k row reads:

- **Unquantized peak at 200 k is 51.28 GB, not the 37 GB a naïve KV-cache
  extrapolation from 65 k predicts.** The extra ~14 GB is prefill working set
  — per-chunk K/V projections and SDPA scratch grow with context even with
  fused attention. KV quant does not shrink that scratch (it's bit-width
  independent), so the *savings* track the cache-only math while the
  *absolute* peak does not. This is a point worth getting right on camera.
- **256 k (model max position embeddings) projection.** The cache-only saving
  scales linearly: ~12.6 GB at 256 k. The unquantized peak at 256 k is now
  expected closer to ~62 GB based on the 200 k trajectory, not ~32 GB. Still
  fits a 128 GB machine with margin, but tighter than the earlier reading.

Decode throughput at long context is the steepest honest tradeoff in this
table: 21.9 → 16.7 tok/s at 65 k (−24 %), and 13.2 → 7.8 tok/s at 200 k
(**−41 %**). At 2 k / 8 k / 32 k the decode penalty is 5–10 %. Every
generated token dereferences the quantized cache, so the cost grows with
context length. If the demo is memory-bound (fitting a big context on a
tight working set) the trade is worth it. If the demo is latency-bound on a
mid-length prompt, the 32 k row is the better pitch.

Raw JSON in `benchmark-results/bench_pass1.json`, `bench_pass2_65k.json`,
and `bench_pass3_200k.json`.

## What ships as the demo server

`server.py` exposes OpenAI-compatible `/v1/chat/completions` (streaming +
non-streaming), `/v1/models`, `/health`. Launched with these settings:

```
model               mlx-community/Qwen3.6-27B-4bit
kv_bits             4
kv_group_size       64
quantized_kv_start  0
prefill_step        2048
max_kv_size         unbounded   (hybrid arch keeps KV tiny even unbounded)
```

Streaming SSE is standard `data: {...}\n\n` + `data: [DONE]\n\n`.
Each non-streaming response carries an `x_mlx_metrics` field with
per-request prefill t/s, decode t/s, peak RSS, peak MLX memory — the exact
numbers to read off-camera.

## Anything I am deliberately NOT claiming

1. **Flash attention as a win.** MLX already uses
   `mx.fast.scaled_dot_product_attention`. Not a differentiating lever.
2. **Asymmetric K/V.** Not implemented; not needed for this model at these
   bit-widths. The TurboQuant insight (K precision dominates) is noted in
   code comments and this doc; it would kick in if we went to aggressive
   bit-widths or a more sensitive model.
3. **Dramatic memory savings at short context.** At 2k context, KV-4bit vs
   no-quant is a small fraction of total memory because the cache is
   already tiny for a hybrid model. The KV-quant story is honest at long
   context (32 k saves 1.53 GB, 65 k saves 3.09 GB, 200 k saves 9.90 GB)
   and continues to scale linearly toward the 256 k position-embedding cap.
4. **Raw hardware specs parity.** Julien was on a 128 GB MBP too. Same
   machine, same model, same weights. What this stack is doing differently
   is (a) MLX instead of llama.cpp — Metal-native, unified-memory aware;
   (b) KV quantization stacked on top of the 4-bit weights without quality
   loss. Both are real, both are measurable, both are below.
