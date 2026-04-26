"""Perplexity evaluation with optional KV cache quantization.

Based on mlx_lm/perplexity.py, but uses chunked forward passes with a prompt
cache so that KV quantization actually exercises the attention path.

The standard mlx_lm.perplexity runs model(seq) in one shot without a cache,
which means KV quantization has no effect — this script fixes that.
"""

import argparse
import math
import time
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import get_total_parameters, load


def load_data(tokenizer, data_path, num_samples, sequence_length):
    args = types.SimpleNamespace(
        hf_dataset={
            "path": data_path,
            "train_split": "train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]
    perm = np.random.permutation(len(dataset)).tolist()
    num_tokens = sequence_length * num_samples if num_samples > 0 else float("inf")
    data = []
    i = 0
    while len(data) < num_tokens:
        tokens, _ = dataset.process(dataset[perm[i]])
        i += 1
        data.extend(tokens)
    data = mx.array(data[: (len(data) // sequence_length) * sequence_length])
    data = data.reshape(-1, sequence_length)
    if num_samples > 0:
        data = data[:num_samples]
    return data


def eval_ppl_chunked(
    model,
    data,
    kv_bits=None,
    kv_group_size=64,
    quantized_kv_start=0,
    prefill_step=512,
):
    """Chunked PPL with a prompt cache.

    For each sample of length L, processes tokens in chunks of prefill_step.
    Between chunks, calls maybe_quantize_kv_cache so later chunks attend over
    quantized KV — this is the same pattern mlx_lm.generate uses internally.
    """
    all_losses = []
    num_samples = len(data)

    for s in range(num_samples):
        sample = data[s]  # (seq_len,)
        seq_len = sample.shape[0]
        cache = make_prompt_cache(model)

        sample_losses = []
        for start in range(0, seq_len, prefill_step):
            end = min(start + prefill_step, seq_len)
            chunk = sample[start:end][None, :]  # (1, chunk_len)
            logits = model(chunk, cache=cache).astype(mx.float32)  # (1, chunk_len, V)

            # logits at chunk position i predicts token at absolute position start+i+1.
            # For chunks where end < seq_len: every position in the chunk has a target.
            # For the final chunk (end == seq_len): drop the last position (no target).
            if end < seq_len:
                targets = sample[start + 1 : end + 1]
                chunk_losses = nn.losses.cross_entropy(
                    logits[0], targets, reduction="none"
                )
            else:
                targets = sample[start + 1 : end]
                chunk_losses = nn.losses.cross_entropy(
                    logits[0, :-1], targets, reduction="none"
                )

            mx.eval(chunk_losses)
            sample_losses.append(chunk_losses)

            if kv_bits is not None:
                maybe_quantize_kv_cache(
                    cache, quantized_kv_start, kv_group_size, kv_bits
                )

            # Discard logits, keep losses
            del logits

        all_losses.append(mx.concatenate(sample_losses))
        del cache
        mx.clear_cache()

        print(f"  Sample {s + 1}/{num_samples} done", end="\r", flush=True)

    print()
    all_losses = mx.concatenate(all_losses)
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    num_tokens = all_losses.size
    standard_error = std_dev / math.sqrt(num_tokens)
    standard_error_ppl = ppl * standard_error
    return ppl, standard_error_ppl, num_tokens


def main():
    parser = argparse.ArgumentParser(description="Chunked PPL with optional KV quant")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--data-path", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--prefill-step", type=int, default=512)
    args = parser.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)
    print(f"Model loaded: {get_total_parameters(model) / 1e6:.1f}M parameters")

    print(f"\nLoading dataset (seq_len={args.sequence_length}, num_samples={args.num_samples})...")
    data = load_data(tokenizer, args.data_path, args.num_samples, args.sequence_length)
    print(f"  Loaded {len(data)} samples")

    kv_desc = (
        "no KV quant"
        if args.kv_bits is None
        else f"kv_bits={args.kv_bits} group={args.kv_group_size} start={args.quantized_kv_start}"
    )
    print(f"\nEvaluating PPL (chunked, prefill_step={args.prefill_step}, {kv_desc})...")

    start_time = time.time()
    ppl, se, ntok = eval_ppl_chunked(
        model,
        data,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step=args.prefill_step,
    )
    eval_time = time.time() - start_time

    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Config:           {kv_desc}")
    print(f"Sequence length:  {args.sequence_length}")
    print(f"Samples:          {args.num_samples}")
    print(f"Tokens scored:    {ntok}")
    print(f"Perplexity:       {ppl:.3f} +/- {se:.3f}")
    print(f"Eval time:        {eval_time:.1f}s")
    print(f"Peak MLX memory:  {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
