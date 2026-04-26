# PHASE2_PLAN.md — full sovereignty matrix

The contract for what Phase 2 will deliver. Time estimates are anchored
to Phase 1 measured prefill scaling, not vibes. Read the budget caveat
in §3 before assuming the full matrix completes in one run.

## 1. Inputs

| | model | repo | on-disk | status |
|---|---|---|---|---|
| Dense | Qwen3.6-27B-4bit | `mlx-community/Qwen3.6-27B-4bit` | 15.1 GB | cached |
| MoE   | Qwen3.6-35B-A3B-4bit | `mlx-community/Qwen3.6-35B-A3B-4bit` | 20.4 GB | downloading (~16.5 / 20.4 GB at plan time) |

Architecture flags worth recording up front:
- Both models are multimodal-tagged HF repos. Dense Qwen3.6-27B is
  `Qwen3_5MoeForConditionalGeneration` — wait, the dense model is
  `Qwen3_5VLForConditionalGeneration` and the MoE here is
  `Qwen3_5MoeForConditionalGeneration` (config field). MLX-LM is happy
  with both via its `qwen3_5_*.py` model files; weights are loaded
  under the `language_model.*` prefix and the vision tower is ignored.
- The Phase 1 dense benchmark already runs as text-only (per
  BENCHMARK.md "vision tower; only language_model is loaded"). MoE is
  expected to behave identically because mlx-lm's `qwen3_5_moe.py`
  module exists and the weight tree has the same `language_model.*`
  shape. Validation step (§4.B) confirms before any matrix runs.
- The HF README for the MoE repo recommends `mlx_vlm.generate`. We are
  going to load it via `mlx_lm.utils.load` instead, exactly as we did
  for the dense 27B. If that fails, fall back path is documented in §6.

Hardware: M4 Max, 128 GB unified memory, macOS 25.5.0. mlx-lm 0.31.3
running under uv-managed Python at
`/Users/tomaxberg/.local/share/uv/tools/mlx-lm/bin/python3`.

## 2. The matrix

Dense (existing data covers 2k/8k/32k/65k/200k):

| ctx | kv=none | kv=4 |
|---:|:---:|:---:|
| 1,000,000 | NEW | NEW |

MoE (all new):

| ctx | kv=none | kv=4 |
|---:|:---:|:---:|
| 16,384 | NEW | NEW |
| 32,768 | NEW | NEW + PPL validation |
| 65,536 | NEW | NEW |
| 200,000 | NEW | NEW |
| 1,000,000 | NEW | NEW |

Total new bench cells: **12**. Plus **2 PPL evals** on MoE @ 32k (one
no-quant baseline, one symmetric 4-bit) using `scripts/ppl_eval.py`.

Same harness as Phase 1: `scripts/bench.py`, greedy decode 32 new
tokens per cell, prefill_step=2048, kv_group_size=64,
quantized_kv_start=0, `mx.reset_peak_memory()` between runs.

## 3. Time budget — honest estimate

Phase 1 measured prefill scaling (dense, kv=none):

| ctx | prefill_secs |
|---:|---:|
| 65,536 | 388 |
| 200,000 | 1,911 |

Implied scaling: `prefill_secs ≈ k · ctx^1.42` between 65k and 200k.
Extrapolating to **1M dense unquant**: `1911 × (1M/200k)^1.42 ≈ 17,000–22,000 s`
→ **~5–6 hours per run**. kv-4bit ~10 % slower in prefill at long ctx.

| Run | Estimated wall time |
|---|---:|
| Dense 1M, kv=none | 5–6 h |
| Dense 1M, kv=4 | 5.5–6.5 h |
| MoE PPL @ 32k (×2) | ~30 min |
| MoE 16k–65k matrix (6 cells) | ~1.5 h |
| MoE 200k matrix (2 cells) | ~3 h |
| MoE 1M matrix (2 cells) | 10–14 h (less certain — depends on MoE prefill profile) |
| **Total** | **~26–32 h** |

**This blows past the 10-hour wall-clock budget by 2.5–3×.** Documented
clearly so it isn't a surprise. Two priority modes:

### 3a. Priority order (worst case — abort at 10 h)
Highest-value cells first; lower-value cells get dropped if we hit the
ceiling.

1. Dense 1M kv=4 (likely-fits headline). ~6 h.
2. MoE PPL validation @ 32k (gates the "no PPL regression" claim). ~30 min.
3. MoE 200k both kv (anchors against existing dense 200k row). ~3 h.
4. MoE 32k + 64k both kv (drives 64GB / 32GB tier story). ~1 h.
5. Dense 1M kv=none (does it OOM? data either way). ~6 h.
6. MoE 16k both kv (low marginal value if 32k already shows the floor). ~10 min.
7. MoE 1M (nice-to-have headline if hours remain). ~10–14 h.

If steps 1–4 land in ~10 h, that already fully supports the post: the
hardware-tier matrix has every (model, context-class) cell needed.
Steps 5–7 are upside.

### 3b. Stop conditions
- **10 h wall clock**: finalize, write up partial matrix, mark missing
  cells "deferred — see PHASE2_PLAN §3a."
- **OOM**: log, continue to next cell. OOM is data, not failure.
- **Two consecutive non-OOM crashes on the same cell**: mark "failed —
  see log," skip that cell, continue.

## 4. Verification gates

### 4.A — Before starting any benchmark
- Both models loadable via `mlx_lm.utils.load`. Dense already proven in
  Phase 1; MoE confirmed in §4.B.
- `mx.get_peak_memory()` resets between runs (already in bench.py).
- No leftover server processes consuming GPU memory.

### 4.B — Before MoE matrix
1. `mlx_lm.utils.load("mlx-community/Qwen3.6-35B-A3B-4bit")` returns a
   model object. If it raises (e.g. unsupported config field or missing
   keys in the `language_model.*` namespace), abort MoE matrix and
   document in BENCHMARK_FULL.md.
2. Generate one short prompt end-to-end (`mlx_lm.generate`) to confirm
   sane output. If it produces gibberish, abort and document.
3. PPL validation @ 32k both kv settings via `scripts/ppl_eval.py`. The
   MoE PPL doesn't have to match dense; what we're validating is that
   `kv=4` PPL ≈ `kv=none` PPL on MoE — same property as dense. If the
   delta is > ~2 %, the "no regression" claim doesn't hold for MoE and
   we say so explicitly in BENCHMARK_FULL.md.

## 5. Hardware-tier recommendation methodology

For each tier, "usable" memory = total − 10 GB working-set headroom
(OS, browser, IDE, the demo client itself).

| Tier | Total | Usable for MLX |
|---|---:|---:|
| 128 GB | 128 | 118 |
| 64 GB | 64 | 54 |
| 32 GB | 32 | 22 |

A measured cell with `peak_mlx_gb ≤ usable(T)` "fits" tier T. The
recommendation per tier is the largest (model, context) combo that
fits, preferring kv=4 when both kv settings fit (lower memory ceiling
= more headroom for the demo client, browser tab, etc).

Caveat surfaced in BENCHMARK_FULL.md: peak measurements come from the
M4 Max 128 GB. Smaller machines may show different MLX allocator
behavior; the 32 GB-tier claim is a prediction until Tom validates on
the m5-32gb machine (§6 ships the exact command).

## 6. Execution order — concrete

**Now (download in flight):**
1. Commit this plan.
2. Kick off **dense 1M kv=4** in tool-background. Monitor stdout for the
   per-run JSON line.

**During dense 1M run (~6 h):**
3. Watch for MoE download completion.
4. When MoE shards finalize (no `.incomplete`), run `mlx_lm.utils.load`
   smoke test in a *separate Python process* (not the bench process —
   we don't want to perturb the running 1M peak measurement).
5. If load smoke passes, run MoE PPL validation in the same separate
   shell on a fresh process while dense 1M continues.

**After dense 1M kv=4 completes:**
6. Kick off **dense 1M kv=none** — this is the OOM check.
7. If kv=none OOMs cleanly, log the OOM and proceed.
8. Kick off MoE matrix in priority order (§3a steps 3–6).

**Final synthesis:**
9. Write `BENCHMARK_FULL.md` with the full table, hardware-tier matrix,
   tradeoffs, recommendations, and the 32GB validation command for Tom.
10. Commit per-step as runs complete (one commit per major step).
11. Print suggested `git remote add` + push commands for Tom.

## 6.5 Architecture facts (read after loading the MoE config)

Pulled from `text_config` in the MoE repo and the Phase 1 BENCHMARK.md:

| | Dense 27B | MoE 35B-A3B |
|---|---:|---:|
| Total layers | 64 | 40 |
| Full-attention layers (1 in N) | 16 (N=4) | 10 (N=4) |
| KV heads | 4 | **2** |
| Head dim | 256 | 256 |
| Per-token KV cache (full-attn layers) | 64 KB | **20 KB** |
| max_position_embeddings | 262,144 | 262,144 |
| Experts (active / total) | n/a | 8 / 256 |
| Active params per token | 26.9 B | 3 B |

Implications:

- **MoE per-token KV cache is 3.2× smaller than dense.** At 200k context
  the MoE cache is ~4 GB vs dense's ~13 GB. At 1M context it's ~20 GB
  vs dense's ~64 GB. The hardware-tier matrix will likely show MoE
  fitting much smaller machines at much longer contexts than dense.
- **MoE 200k kv=4 plausibly fits the 32 GB tier.** Cache ~1 GB +
  scratch ~5 GB + weights ~20 GB = ~26 GB peak prediction. This is the
  cell that, if the prediction holds, anchors the 32 GB tier story.
- **Both models cap at 262,144 RoPE positions.** The 1M runs are
  *memory-mechanics* tests, not quality tests. Memory measurements
  remain valid (allocations are mechanical), but generation at 1M is
  out-of-RoPE-distribution and we will not claim quality at 1M. The
  post should pitch 1M as "this is what the memory budget unlocks if
  the model gains long-context support" rather than "you can do useful
  work at 1M today." This nuance ships in BENCHMARK_FULL.md.

## 7. Assumptions to flag

- **Prefill extrapolation may underestimate.** The N^1.42 fit is from
  two data points (65k, 200k). If actual scaling is closer to N^1.5+,
  1M runs take 8–10 h each instead of 5–6 h.
- **MoE prefill timing unknown.** Sparse FFN suggests faster than 27B
  dense per token, but attention compute is dense and dominates at long
  context. Treating MoE as ~comparable to dense 27B per cell is a
  best-guess; first MoE 32k cell will calibrate the rest.
- **1M unquant likely OOMs.** Dense 200k unquant peak was 51.28 GB.
  KV cache scales linearly: at 1M the cache itself is ~64 GB (16
  full-attn layers × 64 KB/token × 1M tokens). Add prefill scratch
  (~14 GB at 200k, similar absolute at 1M) and 15 GB weights → ~93 GB.
  Tight on 128 GB; could OOM if scratch scales worse than expected.
  This is exactly the OOM data point §6.7 wants to capture.
- **The matrix probably won't complete in one wall-clock window.**
  The plan is designed to produce a defensible BENCHMARK_FULL.md even
  if it stops at the 10-hour gate — see §3a priority order.
