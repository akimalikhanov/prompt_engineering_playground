import os, time, json
from typing import Generator, Optional, Literal
import openai
from utils.sse import sse_pack
from config.load_models_config import _load_models_config

def _unified_openai_api_call(
    model: str,
    messages,
    params: dict,
    stream: bool,
    api_key: str,
    base_url: str,
    stream_mode: Literal["raw","sse"]="raw",
    sse_event: str = "message",
    send_done: bool = False,               # optional status event
    include_metrics: bool = True,           # add TTFT and latency to final event
    max_retries: int = 0
):
    """
    Chat completion via OpenAI-compatible API (OpenAI, vLLM, etc).

    Works with OpenAI and any OpenAI-compatible server by optionally providing
    a base_url.

    Args:
        model (str): Model name to use
        messages: Message content or list of messages
        params (dict): temperature, top_p, max_tokens, seed
        stream (bool): If True, returns streaming response generator
        api_key (str): API key
        base_url (str|None): Optional OpenAI-compatible base URL
        stream_mode (str): Stream format
            - 'raw' -> yields plain text chunks
            - 'sse' -> yields SSE bytes (text/event-stream), ends with metrics
        sse_event (str): Event name for SSE text chunks (default: "message")
        send_done (bool): If True, send a final "done" status event in SSE mode
        include_metrics (bool): If True, add TTFT, latency, tokens, and cost
                                to the final metrics event

    Streaming:
        - Requests stream_options={"include_usage": True} so the API sends a
          final usage chunk.
        - Yields text chunks as they arrive.
        - Emits a final metrics event containing prompt_tokens,
          completion_tokens, cost_usd, ttft_ms, and latency_ms.

    Non-streaming:
        - Returns a dict with:
            {
              "text": str,
              "usage": {"prompt_tokens": int, "completion_tokens": int},
              "metrics": {"ttft_ms": float|None, "latency_ms": float,
                          "model": str, "cost_usd": float|None}
            }

    Returns:
        If stream=True: Generator yielding streaming text chunks
        If stream=False: Dict containing response text, usage, and metrics
    """

    client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)

    normalized_messages = (
        [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    )
    kwargs = {
        "model": model,
        "messages": normalized_messages,
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "seed": params.get("seed"),
        "stream": stream,
    }
    # Some OpenAI-compatible providers balk at 'seed' (e.g., Gemini via proxy)
    if "gemini" in model:
        kwargs.pop("seed", None)

    # If streaming, request usage at end-of-stream
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    res = client.chat.completions.create(**kwargs)

    # --- helpers ------------------------------------------------------------
    def _iter_text(chunks):
        """Yield textual deltas only; ignore non-text chunks."""
        for ch in chunks:
            # ch.choices[0].delta.content is the streamed text piece (may be None)
            try:
                delta = ch.choices[0].delta
                piece = getattr(delta, "content", None)
            except Exception:
                piece = None
            if piece:
                yield piece, ch
            else:
                # still forward the chunk (with None text) so we can inspect usage
                yield None, ch

    def _lookup_pricing(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute USD cost for a given model and token usage."""
        cfg = _load_models_config()
        for m in cfg.get("models", []):
            if m.get("id") == model_id or m.get("model_name") == model_id:
                pricing = m.get("pricing", {})
                in_rate = pricing.get("input", 0.0)   # $ per 1k prompt tokens
                out_rate = pricing.get("output", 0.0) # $ per 1k completion tokens
                cost = (prompt_tokens or 0) * in_rate / 1000.0 \
                    + (completion_tokens or 0) * out_rate / 1000.0
                return round(cost, 6)
        return None  # model not found

    # --- streaming path -----------------------------------------------------
    if stream:
        if stream_mode == "raw":
            def _iter_raw() -> Generator[str, None, None]:
                ttft = None
                chunk_count = 0
                prompt_tokens = None
                completion_tokens = None
                total_tokens= None

                for piece, ch in _iter_text(res):
                    # mark TTFT on first chunk that arrives (text or usage)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    # collect usage if/when the final usage chunk arrives
                    if getattr(ch, "usage", None):
                        try:
                            prompt_tokens = ch.usage.prompt_tokens
                            completion_tokens = ch.usage.completion_tokens
                            total_tokens = ch.usage.total_tokens
                        except Exception:
                            pass
                    if piece:
                        chunk_count += 1
                        yield piece

                total = time.perf_counter() - t0
                if include_metrics:
                    cost_usd = _lookup_pricing(model, prompt_tokens, completion_tokens)
                    metrics = {
                        "metrics": {
                            "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                            "latency_ms": round(total * 1000, 1),
                            "chunks": chunk_count,
                            "model": model,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens, 
                            "tokens_per_chunk": (
                                round((completion_tokens or 0) / (chunk_count or 1), 3)
                                if (completion_tokens is not None) else None
                            ),
                            "cost_usd": cost_usd,
                        }
                    }
                    # emit a newline + JSON so curl clients can read it cleanly
                    yield "\n" + json.dumps(metrics) + "\n"
            return _iter_raw()

        # SSE mode
        def _iter_sse() -> Generator[bytes, None, None]:
            ttft = None
            chunk_count = 0
            prompt_tokens = None
            completion_tokens = None
            total_tokens= None

            for piece, ch in _iter_text(res):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                # capture usage if present (arrives as the last/penultimate chunk)
                if getattr(ch, "usage", None):
                    try:
                        prompt_tokens = ch.usage.prompt_tokens
                        completion_tokens = ch.usage.completion_tokens
                        total_tokens = ch.usage.total_tokens
                    except Exception:
                        pass
                if piece:
                    chunk_count += 1
                    yield sse_pack(piece, event=sse_event)

            total = time.perf_counter() - t0
            if send_done:
                yield sse_pack("done", event="status")
            if include_metrics:
                cost_usd = _lookup_pricing(model, prompt_tokens, completion_tokens)
                payload = {
                    "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                    "latency_ms": round(total * 1000, 1),
                    "chunks": chunk_count,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, 
                    "tokens_per_chunk": (
                        round((completion_tokens or 0) / (chunk_count or 1), 3)
                        if (completion_tokens is not None) else None
                    ),
                    "cost_usd": cost_usd
                }
                yield sse_pack(json.dumps(payload), event="metrics")
        return _iter_sse()

    # --- non-streaming path -------------------------------------------------
    msg = res.choices[0].message
    total = time.perf_counter() - t0

    usage = getattr(res, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    return {
        "text": (msg.content or ""),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "metrics": {
            "ttft_ms": None,  # not meaningful for non-streaming
            "latency_ms": round(total * 1000, 1),
            "model": model,
        }
    }