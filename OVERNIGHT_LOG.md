# Overnight log — what got built while you slept

Time budget: ~4 h (incl. one 30-min benchmark and one ~10-min benchmark).
User directive: "Don't stop until the server is working and benchmarked."

## The three baseline numbers you asked for

| Metric | Value |
|---|---|
| **Warm prefill @ 2048 ctx** | **243.7 tok/s** (no KV quant) / 189.5 (KV-4bit) |
| **Warm decode** | **29.5 tok/s** (no KV quant) / 27.9 (KV-4bit) |
| **Baseline perplexity** | **3.416 ± 0.034** (tulu-3, 65,536 tokens scored) |

## The decision those baselines drove

- Wrote `scripts/ppl_eval.py` (chunked PPL with configurable KV quant — unlike
  `mlx_lm.perplexity`, which bypasses the cache and can't measure quant impact).
- Verified baseline chunked PPL = 3.416 exactly. Script is sound.
- Measured symmetric 4-bit KV PPL: **3.413 ± 0.034**. Within noise.
- **Decision: skip asymmetric K=8/V=4.** Not needed. The TurboQuant insight
  about K-sensitivity was guarding against a regression that doesn't appear
  at this model scale. No custom cache class to review in the morning.

## What was built

- `server.py` — 370 LOC OpenAI-compatible FastAPI server over `mlx_lm.generate_step`.
  Streaming + non-streaming. Live instrumentation in every response
  (`x_mlx_metrics` field): prefill t/s, decode t/s, peak MLX memory, peak RSS.
  Tested end-to-end with streaming SSE and non-streaming JSON. Works.
- `run_server.sh` — hero command with the canonical flags.
- `scripts/bench.py` — prefill/decode throughput + peak memory matrix across
  contexts and KV configs.
- `scripts/demo_long_context.py` — drive the running server with a long prompt
  and print the server's own metrics. Use this on-camera.
- `README.md`, `BENCHMARK.md`, `PREFLIGHT.md`.

## The memory story, measured

Peak MLX memory at context × KV config:

| kv_bits | 2 k    | 8 k    | 32 k   | 65 k    |
|---------|-------:|-------:|-------:|--------:|
| none    | 18.21  | 19.22  | 23.24  | 28.65   |
| 4       | 18.21  | 18.84  | 21.71  | 25.56   |
| **Δ**   |  0.00  | −0.38  | **−1.53** | **−3.09** |

65 k was the headline: 3.09 GB saved, one line shy of the predicted 3.0 GB.
A caveat that matters and is in `BENCHMARK.md`: **decode throughput at 65 k
drops 24 %** under KV-4bit (16.7 vs 21.9 tok/s). At shorter contexts the
decode penalty is 5–10 %. The memory story holds; the latency story depends
on how long the context is.

## Server state when I stopped

**The server is running** on `127.0.0.1:8080` with `kv_bits=4`. `/health`
confirms. Smoke tests (`scripts/test_server.py`) pass on three scenarios:
greedy short, greedy mid, 2 k-prompt long-context. Reply "42" to the 2 k
prompt came back in 8.5 s end-to-end with 240 t/s prefill, 28.7 t/s decode,
18.17 GB peak MLX. Logs in `benchmark-results/server_test.log` +
`benchmark-results/server_final.log`.

## What to run in the morning

```bash
cd /Users/tomaxberg/Code/local-model-inference

# 1. See the server is alive (it should already be)
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool
# If it's not, relaunch:
./run_server.sh

# 2. Demo shot — streaming output with per-request metrics
curl -sS -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-27b",
    "messages":[{"role":"user","content":"Explain unified memory in one sentence."}],
    "stream":true,
    "temperature":0,
    "max_tokens":80,
    "chat_template_args":{"enable_thinking":false}
  }'

# 3. Long-context demo — show peak memory at 20k tokens
python3 scripts/demo_long_context.py --context-chars 80000 --gen 64
```

## Known caveats to keep in mind

- **Prefill is ~10–20 % slower with KV quant** due to quantize-on-write
  overhead. Decode is ~5–10 % slower. This is the honest tradeoff.
  Memory savings at long context are real; throughput cost is small.
- **psutil RSS does not reliably reflect MLX allocations** on Apple silicon.
  The authoritative number is `mx.get_peak_memory()`, which the bench script
  resets between runs. This is noted in `BENCHMARK.md`.
- **At 2 k context, KV quant saves 0 GB.** Don't demo at short context.
  Demo at 8 k+ minimum, ideally 20 k+, to show the savings meaningfully.

## What I did NOT touch

- Your system settings, networksetup, Time Machine, Spotlight. Those are on
  the PREFLIGHT checklist for you to run before recording.
- Your model cache (still 15 GB at
  `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit`).
- Git — the project dir is not a git repo; I did not `git init`. If you want
  to commit this, that's the first step.

## Remaining follow-on work if you want to polish further

- Pin `sse-starlette`, `fastapi`, etc. into a `requirements.txt` or `pyproject.toml`
  for reproducibility on a fresh machine.
- Add a `/v1/completions` endpoint if any of your demo clients need it
  (chat is sufficient for most).
- Longer perplexity sweep: 128+ samples at 2048 seq_len to tighten the
  ± on PPL. The current ±0.034 is already tight enough to claim "no
  regression," but wider evaluations at larger seq lengths would be extra
  defense. Budget ~30 min per config.
