"""Smoke-test the running server against a short list of scenarios.

Exits 0 on success, 1 on first failure. Prints per-scenario timing and
the server-reported metrics. Good as a pre-flight check after restarting
the server.

Usage:
    python scripts/test_server.py                      # defaults to 127.0.0.1:8080
    python scripts/test_server.py --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def health(host, port):
    url = f"http://{host}:{port}/health"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def chat(host, port, **kwargs):
    url = f"http://{host}:{port}/v1/chat/completions"
    data = json.dumps(kwargs).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode()
    return json.loads(body), time.perf_counter() - t0


def run(host, port):
    print(f"→ GET /health on {host}:{port}")
    h = health(host, port)
    print(json.dumps(h, indent=2))
    assert h.get("status") == "ok", "health not ok"
    print()

    scenarios = [
        {
            "name": "greedy short",
            "req": {
                "model": "qwen3.6-27b",
                "messages": [{"role": "user", "content": "Say exactly: ok"}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 4,
                "chat_template_args": {"enable_thinking": False},
            },
            "expect_contains": "ok",
        },
        {
            "name": "greedy mid",
            "req": {
                "model": "qwen3.6-27b",
                "messages": [{"role": "user", "content": "In one sentence: what is unified memory?"}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 80,
                "chat_template_args": {"enable_thinking": False},
            },
        },
        {
            "name": "long context",
            "req": {
                "model": "qwen3.6-27b",
                "messages": [{
                    "role": "user",
                    "content": "Context: " + ("word " * 2000) +
                    "\n\nQuestion: Reply with exactly the number 42.",
                }],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 8,
                "chat_template_args": {"enable_thinking": False},
            },
        },
    ]

    for s in scenarios:
        print(f"→ {s['name']}")
        body, elapsed = chat(host, port, **s["req"])
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        metrics = body.get("x_mlx_metrics", {})
        print(f"  wall: {elapsed:.2f}s")
        print(f"  usage: {usage}")
        if metrics:
            prefill = metrics.get("prefill_tps", 0)
            decode = metrics.get("decode_tps", 0)
            peak = metrics.get("peak_mlx_gb", 0)
            print(f"  prefill: {prefill:.1f} t/s, decode: {decode:.1f} t/s, peak MLX: {peak:.2f} GB")
        print(f"  reply: {content[:140]!r}")
        if "expect_contains" in s:
            if s["expect_contains"].lower() not in content.lower():
                print(f"  FAIL: expected '{s['expect_contains']}' in reply")
                return 1
        print("  OK")
        print()

    print("All scenarios passed.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    sys.exit(run(args.host, args.port))


if __name__ == "__main__":
    main()
