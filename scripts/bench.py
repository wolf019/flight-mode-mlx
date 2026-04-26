"""Benchmark matrix: prefill/decode throughput and peak memory across
context lengths and KV configurations.

Usage:
    python scripts/bench.py --model mlx-community/Qwen3.6-27B-4bit

By default runs the matrix:
    contexts   = [2048, 8192, 32768, 65536]
    kv_configs = [None, 4, 8]   # None = no KV quant

For each (context, kv_bits) it reports:
    prefill tok/s   (steady-state, after warmup)
    decode tok/s    (128 new tokens)
    peak_mlx_gb     (via mx.get_peak_memory, reset between runs)
    rss_gb          (end-of-run OS RSS — cumulative; for sanity only)

Writes JSON lines to stdout and a full JSON report to
benchmark-results/bench.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


def build_cache(model, max_kv_size=None):
    cache = make_prompt_cache(model)
    if max_kv_size is not None:
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                cache[i] = RotatingKVCache(max_size=max_kv_size, keep=4)
    return cache


def synth_prompt(tokenizer, n_tokens: int) -> mx.array:
    """Build a deterministic synthetic prompt of the requested length.

    Uses a fixed random seed so the same prompt is used across runs — critical
    for comparing prefill times fairly.
    """
    vocab = getattr(tokenizer, "vocab_size", None) or 200000
    rng = np.random.RandomState(42)
    ids = rng.randint(1, min(vocab, 200000), n_tokens).tolist()
    return mx.array(ids)


def run_one(
    model,
    tokenizer,
    prompt_len: int,
    gen_len: int,
    kv_bits,
    kv_group_size,
    quantized_kv_start,
    prefill_step,
    max_kv_size,
):
    prompt = synth_prompt(tokenizer, prompt_len)
    sampler = make_sampler(temp=0.0)

    # Reset peak tracker so this run's peak is clean.
    mx.clear_cache()
    mx.reset_peak_memory()

    cache = build_cache(model, max_kv_size)

    t_start = time.perf_counter()
    first_time = None
    count = 0
    for token, _ in generate_step(
        prompt=prompt,
        model=model,
        max_tokens=gen_len,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=prefill_step,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    ):
        if first_time is None:
            first_time = time.perf_counter()
        count += 1
    t_end = time.perf_counter()

    prefill_secs = (first_time - t_start) if first_time else 0.0
    decode_secs = (t_end - first_time) if first_time else 0.0
    prefill_tps = prompt_len / prefill_secs if prefill_secs > 0 else 0.0
    decode_tps = count / decode_secs if decode_secs > 0 else 0.0

    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    peak_mlx_gb = mx.get_peak_memory() / 1e9

    # Free the cache before next run to make mx.reset_peak_memory meaningful.
    del cache
    mx.clear_cache()

    return {
        "prompt_tokens": prompt_len,
        "gen_tokens": count,
        "prefill_secs": round(prefill_secs, 3),
        "decode_secs": round(decode_secs, 3),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 1),
        "rss_gb": round(rss_gb, 2),
        "peak_mlx_gb": round(peak_mlx_gb, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3.6-27B-4bit")
    p.add_argument(
        "--contexts", type=int, nargs="+",
        default=[2048, 8192, 32768, 65536],
    )
    p.add_argument(
        "--kv-configs", type=str, nargs="+",
        default=["none", "4", "8"],
        help="List of kv_bits values to test. 'none' = no quantization.",
    )
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=0)
    p.add_argument("--prefill-step", type=int, default=2048)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--out", default="benchmark-results/bench.json")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}…", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_secs = time.perf_counter() - t0
    print(f"Loaded in {load_secs:.1f}s", flush=True)

    # Short warmup (outside the matrix) to make sure Metal kernels are compiled.
    print("Warming up kernels…", flush=True)
    _ = run_one(
        model, tokenizer,
        prompt_len=256, gen_len=8,
        kv_bits=None,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step=args.prefill_step,
        max_kv_size=args.max_kv_size,
    )
    # Warmup for the quantized-kernel path too.
    _ = run_one(
        model, tokenizer,
        prompt_len=256, gen_len=8,
        kv_bits=4,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step=args.prefill_step,
        max_kv_size=args.max_kv_size,
    )
    print("Warmup done.", flush=True)

    results = []
    for kv_spec in args.kv_configs:
        kv_bits = None if kv_spec.lower() == "none" else int(kv_spec)
        for ctx in args.contexts:
            label = f"kv={kv_spec} ctx={ctx}"
            print(f"\n=== {label} ===", flush=True)
            r = run_one(
                model, tokenizer,
                prompt_len=ctx,
                gen_len=args.gen_tokens,
                kv_bits=kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
                prefill_step=args.prefill_step,
                max_kv_size=args.max_kv_size,
            )
            r["config"] = {
                "kv_bits": kv_bits,
                "context": ctx,
                "gen_tokens": args.gen_tokens,
                "kv_group_size": args.kv_group_size,
                "quantized_kv_start": args.quantized_kv_start,
                "prefill_step": args.prefill_step,
                "max_kv_size": args.max_kv_size,
            }
            results.append(r)
            print(json.dumps(r))

    # Table summary
    print("\n" + "=" * 80)
    print(f"{'kv_bits':>8} {'ctx':>8} {'prefill t/s':>12} {'decode t/s':>12} {'peak MLX GB':>12} {'RSS GB':>10}")
    print("-" * 80)
    for r in results:
        c = r["config"]
        kv = c["kv_bits"] if c["kv_bits"] is not None else "none"
        print(
            f"{str(kv):>8} {c['context']:>8} {r['prefill_tps']:>12.1f} "
            f"{r['decode_tps']:>12.1f} {r['peak_mlx_gb']:>12.2f} {r['rss_gb']:>10.2f}"
        )

    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "load_secs": round(load_secs, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
