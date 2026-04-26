# Pre-Flight Checklist — Demo Recording

Run through this before hitting record. Each item should take under a minute.
The goal: a clean, reproducible run that will hold up when someone rewinds
the tape.

**Default model for the demo is MoE Qwen3.6-35B-A3B-4bit** (Phase 2
recommendation; see `BENCHMARK_FULL.md` for why). Dense Qwen3.6-27B-4bit
is the apples-to-apples llama.cpp comparison point and remains
fully supported — paths in §7, §8, §9 list both.

## 1. Network (the narrative requires airplane mode)

```bash
# Airplane mode equivalent on macOS
networksetup -setairportpower en0 off
blueutil --power 0        # brew install blueutil if missing
# Optional: fully disable all network interfaces
sudo ifconfig en0 down 2>/dev/null; sudo ifconfig en1 down 2>/dev/null
```

Verify offline:
```bash
curl -sS --max-time 2 https://example.com/ >/dev/null && echo "STILL ONLINE" || echo "offline"
```

When you restore network, reverse with `networksetup -setairportpower en0 on`.

## 2. Power

Plug in the adapter. M-series throttles GPU on battery for sustained loads —
this is visible in decode tok/s and ruins long-context prefill.

```bash
pmset -g batt   # should show "AC Power"
```

## 3. Display + sleep

Prevent screen dimming and sleep during the recording. Do NOT disable the
display itself — you need it visible.

```bash
caffeinate -dimsu &   # keep running in a terminal tab
```

Kill with `kill %1` after recording.

## 4. macOS noise sources

```bash
# Pause Spotlight indexing of the model cache
sudo mdutil -i off ~/.cache/huggingface

# Pause Time Machine
sudo tmutil disable

# Close visibly-active apps that run background tasks
osascript -e 'tell application "Mail" to quit'
osascript -e 'tell application "Slack" to quit'
osascript -e 'tell application "Google Chrome" to quit' 2>/dev/null

# Stop any stray Python / MLX processes
pkill -f 'python.*mlx' 2>/dev/null
pkill -f 'ollama'     2>/dev/null
pkill -f 'lm-studio'  2>/dev/null
```

Re-enable after recording:
```bash
sudo mdutil -i on ~/.cache/huggingface
sudo tmutil enable
```

## 5. Terminal

- Font size ≥ 16pt, white-on-black, bold. Full-screen the window.
- Clear buffer (`printf '\033[2J\033[H'`) before each run.
- `unset HISTFILE` so secrets / mistakes don't persist.
- One terminal tab = server, one tab = curl client, one tab = `htop`.

## 6. Process + version sanity

```bash
# mlx-lm version
python3 -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"

# Python version
python3 --version

# Hardware attestation
system_profiler SPHardwareDataType | grep -E 'Chip|Memory'
sw_vers
```

Expected on this machine: M4 Max, 128 GB unified memory, macOS 25.5.0,
mlx_lm ≥ 0.31.3.

## 7. Model attestation

```bash
# MoE (default for the demo)
ls -la ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*/
# Expected: 4 safetensors shards, ~20.4 GB total.
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/

# Dense (Phase 1 / llama.cpp comparison)
ls -la ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/*/
# Expected: 3 safetensors shards, ~15.1 GB total.
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/
```

## 8. Warm the model once, off-camera

Every `mlx_lm` invocation compiles Metal kernels on first use. Do one dry
run before recording so the recorded run shows steady-state numbers, not
compilation overhead.

```bash
# MoE (default)
mlx_lm.generate \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt "say ok" \
  --max-tokens 4 \
  --temp 0 \
  --chat-template-config '{"enable_thinking": false}'

# Dense (only if you'll record the dense path)
mlx_lm.generate \
  --model mlx-community/Qwen3.6-27B-4bit \
  --prompt "say ok" \
  --max-tokens 4 \
  --temp 0 \
  --chat-template-config '{"enable_thinking": false}'
```

## 9. Server launch (the hero command)

```bash
# MoE (default for the demo)
MODEL_PATH=mlx-community/Qwen3.6-35B-A3B-4bit KV_BITS=4 ./run_server.sh

# Or dense (llama.cpp apples-to-apples)
MODEL_PATH=mlx-community/Qwen3.6-27B-4bit KV_BITS=4 ./run_server.sh

# Or fully explicit
python3 server.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --kv-bits 4 \
  --kv-group-size 64 \
  --quantized-kv-start 0 \
  --prefill-step 2048 \
  --warmup-tokens 32 \
  --port 8080
```

Expect to see: `Model loaded in ~3s`, `EOS token ids: [...]`,
`Warmup done in Xs — prefill Y tok/s, decode Z tok/s, peak RSS N GB`.

Expected baseline RSS at warmup:
- MoE: ~20 GB (4-bit weights + minimal cache).
- Dense: ~15 GB.

If RSS at warmup exceeds these by >2 GB, weights are NOT loaded at 4-bit.
Stop and investigate.

## 10. Smoke probe the endpoint

```bash
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool
```

Then a fast, deterministic demo request (model name is informational
in the request, server doesn't validate it):

```bash
curl -sS -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Say: flight mode engaged"}],"stream":true,"temperature":0,"max_tokens":20,"chat_template_args":{"enable_thinking":false}}' \
  2>&1 | head -40
```

## 11. The actual demo arc (what to screen-record)

The MoE arc (default — what the post recommends):

1. Show `pmset -g batt` on AC power and `networksetup` showing Wi-Fi OFF.
2. Clear terminal. Launch server with MoE + kv=4. Wait for the warmup line.
3. Show `/health` with `kv_bits: 4`, `peak RSS ≈ 20 GB` (MoE weights).
4. Run a long prompt at 200k context using
   `python3 scripts/demo_long_context.py --context-chars 800000 --gen 64`.
   Watch the response come back. Narrate the `x_mlx_metrics` payload —
   prefill ~390 t/s, decode ~26 t/s, peak ~36 GB. **Decode at 25.8 t/s on
   200k context, on a laptop, no network — that's the headline.**
5. Compare: stop server, relaunch with `KV_BITS=` (no quant), run the same
   long prompt. Peak ~39 GB; decode jumps to ~49 t/s. Show that the
   memory savings come at a decode cost — the honest tradeoff.
6. Show PPL parity in `BENCHMARK_FULL.md` §2 (3.659 kv=4 vs 3.703 baseline
   on MoE, same noise property as the dense Phase 1 numbers).

The dense arc (apples-to-apples llama.cpp comparison, secondary):

7. Same flow but with `MODEL_PATH=mlx-community/Qwen3.6-27B-4bit`. Shows
   that this stack works with the model Julien ran — and adds KV-quant
   on top without quality loss. Decode at 200k will be ~8 t/s (sub
   threshold) — frame as "memory math holds for the dense case too,
   but you'd actually deploy the MoE."

## 12. Post-recording cleanup

```bash
pkill -f "python3 server.py"
kill %1                             # caffeinate
networksetup -setairportpower en0 on
blueutil --power 1
sudo mdutil -i on ~/.cache/huggingface
sudo tmutil enable
```

## Things that will ruin the demo (pre-mortem)

- **Thermals.** If the machine has been running other MLX jobs, the soc may
  be hot. Let it cool 5 minutes. A cold decode is ~30 tok/s; throttled it
  drops to 15.
- **Prompt template changes.** Qwen3.6 has thinking by default; the demo
  uses `enable_thinking: false` to keep outputs tight. Forgetting this
  produces multi-paragraph chain-of-thought instead of the expected answer.
- **Process with model leftover.** An earlier server run still holding
  the weights + a new launch = double the baseline RSS (e.g. 40 GB
  instead of 20 GB for MoE, 30 GB instead of 15 GB for dense).
  `lsof -i :8080` and `ps -o rss= -p $(pgrep -f server.py)` catch it.
- **Power source.** Recording on battery silently halves throughput.
- **Kernel warmup on camera.** Always warm once off-camera first.
