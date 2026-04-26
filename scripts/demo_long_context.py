"""Drive the running server with a long synthetic prompt and print metrics.

Usage:
    python scripts/demo_long_context.py --context 32768 --gen 128

Constructs a coherent prompt of approximately the requested token length
(using a repeated passage + a question), POSTs to the server's
/v1/chat/completions with streaming disabled, and pretty-prints the
x_mlx_metrics block — which tells you prefill tok/s, decode tok/s, peak
RSS, and peak MLX memory for THIS request.

The server must already be running on --port.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


PASSAGE = (
    "MLX is Apple's array framework designed for efficient and flexible "
    "machine learning on Apple silicon, where CPU and GPU share unified "
    "memory. This unified memory architecture eliminates the memory "
    "transfer overhead that traditionally dominates inference on discrete "
    "GPUs. Quantizing weights to 4 bits, combined with per-layer cache "
    "management, allows models of 27 billion parameters to run on consumer "
    "laptops while remaining responsive. Sovereignty in local inference is "
    "ultimately a memory story: the right combination of weight "
    "quantization, KV cache quantization, and bounded attention state "
    "turns a hypothetical flight-mode assistant into a useful one. "
)


def build_prompt(target_chars: int) -> str:
    repeats = max(1, target_chars // len(PASSAGE))
    body = PASSAGE * repeats
    return (
        "You are a careful technical writer. The text below is background "
        "context; read it before answering the question at the end.\n\n"
        + body
        + "\n\nQuestion: In one sentence, what is the single tightest "
        "constraint on running large language models on Apple silicon?"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--context-chars", type=int, default=80_000,
                   help="Approximate prompt character count. "
                        "80k chars ~= 20k tokens on Qwen tokenizer. "
                        "800k chars ~= 200k tokens for the MoE long-context demo.")
    p.add_argument("--gen", type=int, default=128,
                   help="max_tokens to request from the server.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--model-name", default="qwen3.6-35b-a3b",
                   help="Model name string in the request (server is lax; this "
                        "just shows up in the response. Use 'qwen3.6-27b' for "
                        "the dense Phase 1 server, 'qwen3.6-35b-a3b' for MoE.")
    args = p.parse_args()

    prompt = build_prompt(args.context_chars)
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.gen,
        "chat_template_args": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    print(f"POST {url}")
    print(f"Prompt char count: {len(prompt):,}")
    print(f"Waiting for server…")

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = resp.read().decode()
    elapsed = time.perf_counter() - t0

    js = json.loads(body)
    content = js["choices"][0]["message"]["content"]
    usage = js.get("usage", {})
    metrics = js.get("x_mlx_metrics", {})

    print(f"\nWall clock: {elapsed:.1f}s")
    print(f"Usage: {usage}")
    if metrics:
        print("\nServer-reported metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:20s} {v:.2f}")
            else:
                print(f"  {k:20s} {v}")

    print("\n--- model output ---")
    print(content[:500])
    if len(content) > 500:
        print(f"... [truncated, full length {len(content)}]")


if __name__ == "__main__":
    main()
