"""Flight-mode local inference server for Qwen3.6-27B-4bit.

OpenAI-compatible /v1/chat/completions over mlx_lm with symmetric 4-bit KV
cache quantization. Minimal surface area, full memory + throughput
instrumentation.

Run:
    python server.py --model mlx-community/Qwen3.6-27B-4bit --kv-bits 4

Then:
    curl -N http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Optional

import mlx.core as mx
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


# ---------------------------------------------------------------------------
# Config

class Config:
    model_path: str
    kv_bits: Optional[int]
    kv_group_size: int
    quantized_kv_start: int
    max_kv_size: Optional[int]
    prefill_step: int
    host: str
    port: int
    warmup_tokens: int

    @classmethod
    def from_args(cls, a: argparse.Namespace) -> "Config":
        c = cls()
        c.model_path = a.model
        c.kv_bits = a.kv_bits
        c.kv_group_size = a.kv_group_size
        c.quantized_kv_start = a.quantized_kv_start
        c.max_kv_size = a.max_kv_size
        c.prefill_step = a.prefill_step
        c.host = a.host
        c.port = a.port
        c.warmup_tokens = a.warmup_tokens
        return c


# ---------------------------------------------------------------------------
# Globals populated at startup

CFG: Config = None  # type: ignore[assignment]
MODEL = None
TOKENIZER = None
EOS_IDS: set[int] = set()
GEN_LOCK = Lock()
LOG = logging.getLogger("server")


# ---------------------------------------------------------------------------
# Memory probes

_PROC = psutil.Process(os.getpid())

def rss_gb() -> float:
    return _PROC.memory_info().rss / 1e9

def mlx_peak_gb() -> float:
    return mx.get_peak_memory() / 1e9


# ---------------------------------------------------------------------------
# Cache helpers

def build_cache():
    """Build a prompt cache for the loaded model.

    Qwen3.6-27B is hybrid: linear-attention layers get ArraysCache, full-
    attention layers get KVCache. If --max-kv-size is set, full-attention
    KVCaches are swapped for RotatingKVCache so the KV footprint is bounded.
    """
    cache = make_prompt_cache(MODEL)
    if CFG.max_kv_size is not None:
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                cache[i] = RotatingKVCache(max_size=CFG.max_kv_size, keep=4)
    return cache


# ---------------------------------------------------------------------------
# Inference

def format_messages(messages: list[dict], chat_template_args: Optional[dict] = None) -> list[int]:
    """Apply the tokenizer's chat template; return token ids."""
    if not TOKENIZER.has_chat_template:
        # Fall back to concatenated roles
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return TOKENIZER.encode(text)
    kwargs = dict(chat_template_args or {})
    return TOKENIZER.apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        add_generation_prompt=True,
        **kwargs,
    )


def run_generation(
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: Optional[float],
    stop_token_ids: set[int],
):
    """Yield (token_id, text_delta, is_eos) tuples.

    Also yields one final `_Stats` object as the last item.
    """
    prompt_tensor = mx.array(prompt_ids)
    cache = build_cache()
    sampler = make_sampler(
        temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p
    )
    processors = make_logits_processors(
        repetition_penalty=repetition_penalty,
    ) if repetition_penalty else None

    t_start = time.perf_counter()
    first_token_time = None
    generated_ids: list[int] = []
    detokenizer = TOKENIZER.detokenizer
    detokenizer.reset()

    gen = generate_step(
        prompt=prompt_tensor,
        model=MODEL,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
        prompt_cache=cache,
        prefill_step_size=CFG.prefill_step,
        kv_bits=CFG.kv_bits,
        kv_group_size=CFG.kv_group_size,
        quantized_kv_start=CFG.quantized_kv_start,
    )
    for token, _logprobs in gen:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if token in stop_token_ids:
            break
        generated_ids.append(token)
        detokenizer.add_token(token)
        text = detokenizer.last_segment
        if text:
            yield token, text, False

    # Flush any remaining text in the detokenizer
    detokenizer.finalize()
    tail = detokenizer.last_segment
    if tail:
        yield None, tail, False

    t_end = time.perf_counter()

    prompt_tokens = len(prompt_ids)
    completion_tokens = len(generated_ids)
    prefill_secs = (first_token_time - t_start) if first_token_time else 0.0
    decode_secs = (t_end - (first_token_time or t_end))
    prefill_tps = prompt_tokens / prefill_secs if prefill_secs > 0 else 0.0
    decode_tps = completion_tokens / decode_secs if decode_secs > 0 else 0.0

    yield _Stats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prefill_secs=prefill_secs,
        decode_secs=decode_secs,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        peak_rss_gb=rss_gb(),
        peak_mlx_gb=mlx_peak_gb(),
    ), None, True


class _Stats:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def as_dict(self):
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# FastAPI

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.0
    max_tokens: int = Field(512, ge=1, le=32768)
    repetition_penalty: Optional[float] = None
    stop: Optional[list[str] | str] = None
    # Passed through to tokenizer.apply_chat_template — e.g. {"enable_thinking": false}
    chat_template_args: Optional[dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, EOS_IDS
    LOG.info("Loading model %s", CFG.model_path)
    t0 = time.perf_counter()
    MODEL, TOKENIZER = load(CFG.model_path)
    load_secs = time.perf_counter() - t0
    LOG.info("Model loaded in %.2fs (RSS %.2f GB, MLX peak %.2f GB)",
             load_secs, rss_gb(), mlx_peak_gb())

    # Collect EOS token ids — Qwen3.6 has multiple.
    eos_ids: set[int] = set()
    tok = TOKENIZER
    for attr in ("eos_token_id", "eos_token_ids"):
        v = getattr(tok, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            eos_ids.update(int(x) for x in v)
        else:
            eos_ids.add(int(v))
    # config.json on Qwen3.6-27B also has a list form
    cfg_eos = getattr(MODEL, "eos_token_id", None)
    if isinstance(cfg_eos, (list, tuple)):
        eos_ids.update(int(x) for x in cfg_eos)
    EOS_IDS = eos_ids
    LOG.info("EOS token ids: %s", sorted(EOS_IDS))

    # Warmup
    if CFG.warmup_tokens > 0:
        LOG.info("Warming up with %d tokens…", CFG.warmup_tokens)
        t0 = time.perf_counter()
        prompt_ids = TOKENIZER.encode("Hello, write a short greeting.")
        stats_seen = None
        for tok_id, text, is_final in run_generation(
            prompt_ids=prompt_ids,
            max_tokens=CFG.warmup_tokens,
            temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
            repetition_penalty=None,
            stop_token_ids=EOS_IDS,
        ):
            if is_final:
                stats_seen = tok_id  # _Stats object
        warm_secs = time.perf_counter() - t0
        if stats_seen is not None:
            LOG.info(
                "Warmup done in %.2fs — prefill %.1f tok/s, decode %.1f tok/s, peak RSS %.2f GB",
                warm_secs, stats_seen.prefill_tps, stats_seen.decode_tps, stats_seen.peak_rss_gb,
            )

    yield
    LOG.info("Shutting down. Final peak RSS %.2f GB, MLX peak %.2f GB",
             rss_gb(), mlx_peak_gb())


app = FastAPI(lifespan=lifespan, title="flight-mode-server", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": CFG.model_path,
        "kv_bits": CFG.kv_bits,
        "kv_group_size": CFG.kv_group_size,
        "quantized_kv_start": CFG.quantized_kv_start,
        "max_kv_size": CFG.max_kv_size,
        "prefill_step": CFG.prefill_step,
        "rss_gb": round(rss_gb(), 2),
        "mlx_peak_gb": round(mlx_peak_gb(), 2),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": CFG.model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


def _openai_chunk(cid: str, created: int, model: str, delta: dict, finish_reason: Optional[str]):
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = req.model or CFG.model_path

    # Build stop ids
    stop_ids = set(EOS_IDS)
    if req.stop:
        stops = [req.stop] if isinstance(req.stop, str) else list(req.stop)
        for s in stops:
            for t in TOKENIZER.encode(s):
                stop_ids.add(int(t))

    # Apply chat template
    try:
        prompt_ids = format_messages(
            [m.model_dump() for m in req.messages],
            chat_template_args=req.chat_template_args,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"chat template failed: {e}")

    if req.stream:
        def event_stream():
            # Acquire inference lock for the duration of the request
            with GEN_LOCK:
                # Role opener
                first = _openai_chunk(
                    cid, created, model_id,
                    delta={"role": "assistant"},
                    finish_reason=None,
                )
                yield f"data: {json.dumps(first)}\n\n"

                stats: Optional[_Stats] = None
                finish_reason = "length"
                for tok_id, text, is_final in run_generation(
                    prompt_ids=prompt_ids,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    min_p=req.min_p,
                    repetition_penalty=req.repetition_penalty,
                    stop_token_ids=stop_ids,
                ):
                    if is_final:
                        stats = tok_id  # _Stats
                        finish_reason = "stop" if stats.completion_tokens < req.max_tokens else "length"
                        break
                    if text:
                        chunk = _openai_chunk(
                            cid, created, model_id,
                            delta={"content": text},
                            finish_reason=None,
                        )
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Closing chunk
                closing = _openai_chunk(
                    cid, created, model_id,
                    delta={},
                    finish_reason=finish_reason,
                )
                yield f"data: {json.dumps(closing)}\n\n"
                yield "data: [DONE]\n\n"

                if stats is not None:
                    LOG.info(
                        "%s | prompt=%d completion=%d | prefill=%.1f t/s decode=%.1f t/s | RSS=%.2f GB MLX=%.2f GB",
                        cid, stats.prompt_tokens, stats.completion_tokens,
                        stats.prefill_tps, stats.decode_tps,
                        stats.peak_rss_gb, stats.peak_mlx_gb,
                    )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path
    with GEN_LOCK:
        content_parts: list[str] = []
        stats: Optional[_Stats] = None
        for tok_id, text, is_final in run_generation(
            prompt_ids=prompt_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            stop_token_ids=stop_ids,
        ):
            if is_final:
                stats = tok_id
                break
            if text:
                content_parts.append(text)

    content = "".join(content_parts)
    finish_reason = "stop" if stats and stats.completion_tokens < req.max_tokens else "length"

    if stats is not None:
        LOG.info(
            "%s | prompt=%d completion=%d | prefill=%.1f t/s decode=%.1f t/s | RSS=%.2f GB MLX=%.2f GB",
            cid, stats.prompt_tokens, stats.completion_tokens,
            stats.prefill_tps, stats.decode_tps,
            stats.peak_rss_gb, stats.peak_mlx_gb,
        )

    usage = {
        "prompt_tokens": stats.prompt_tokens if stats else len(prompt_ids),
        "completion_tokens": stats.completion_tokens if stats else 0,
        "total_tokens": (stats.prompt_tokens + stats.completion_tokens) if stats else len(prompt_ids),
    }

    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "x_mlx_metrics": stats.as_dict() if stats else None,
    })


# ---------------------------------------------------------------------------
# Entry point

def _kv_bits_arg(s: str) -> Optional[int]:
    if s is None or str(s).lower() in ("none", "off", "disable", "disabled", ""):
        return None
    return int(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "mlx-community/Qwen3.6-27B-4bit"))
    parser.add_argument("--kv-bits", type=_kv_bits_arg, default=_kv_bits_arg(os.getenv("KV_BITS", "4")),
                        help="Bits for KV cache quantization. Pass 'none' or '--no-kv-quant' to disable.")
    parser.add_argument("--no-kv-quant", action="store_true",
                        help="Disable KV cache quantization entirely (overrides --kv-bits).")
    parser.add_argument("--kv-group-size", type=int, default=int(os.getenv("KV_GROUP_SIZE", "64")))
    parser.add_argument("--quantized-kv-start", type=int, default=int(os.getenv("QUANTIZED_KV_START", "0")))
    parser.add_argument("--max-kv-size", type=int, default=None, help="Cap KV cache size (rotating). None = unbounded.")
    parser.add_argument("--prefill-step", type=int, default=int(os.getenv("PREFILL_STEP", "2048")))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--warmup-tokens", type=int, default=16, help="Tokens to decode at startup to warm kernels.")
    args = parser.parse_args()
    if args.no_kv_quant:
        args.kv_bits = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    global CFG
    CFG = Config.from_args(args)

    LOG.info(
        "Config: model=%s kv_bits=%s group=%d start=%d max_kv=%s prefill_step=%d host=%s port=%d",
        CFG.model_path, CFG.kv_bits, CFG.kv_group_size, CFG.quantized_kv_start,
        CFG.max_kv_size, CFG.prefill_step, CFG.host, CFG.port,
    )

    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
