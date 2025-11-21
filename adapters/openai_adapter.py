import os, time, json
from typing import Generator, Optional, Literal, Union
from dataclasses import dataclass

import openai

from services.llm_tools import get_tool_callable
from utils.sse import sse_pack
from utils.load_configs import _load_models_config


@dataclass
class ToolStatusMarker:
    """Marker to indicate tool execution status in streaming."""
    tool_names: list


MAX_TOOL_ITERATIONS = 5


def _is_o4_model(model: str) -> bool:
    return model.startswith("o4") or "o4-mini" in model


def _is_o_series_model(model: str) -> bool:
    return model.startswith("o") and (
        "-mini" in model or model.startswith(("o1", "o3", "o4"))
    )


def _supports_seed(model: str) -> bool:
    return (not _is_o_series_model(model)) and ("gemini" not in model)


def _normalize_response_format(response_format):
    if response_format is None:
        return None

    if hasattr(response_format, "model_dump"):
        response_format_dict = response_format.model_dump(exclude_none=True)
    elif isinstance(response_format, dict):
        response_format_dict = response_format.copy()
    else:
        response_format_dict = dict(response_format)

    json_schema_config = response_format_dict.get("json_schema")
    if isinstance(json_schema_config, dict):
        response_format_dict["json_schema"] = {
            k: v for k, v in json_schema_config.items() if v is not None
        }

    if response_format_dict.get("type") == "json_schema":
        json_schema = response_format_dict.setdefault("json_schema", {})
        if isinstance(json_schema, dict) and "strict" not in json_schema:
            json_schema["strict"] = True

    return response_format_dict


def _iter_unique_tool_names(tool_names):
    seen = set()
    for tool_name in tool_names:
        cleaned = (tool_name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned


def _is_tool_loop(recent_tool_calls, current_tool_signatures):
    return (
        len(recent_tool_calls) >= 2
        and recent_tool_calls[-1] == current_tool_signatures
        and recent_tool_calls[-2] == current_tool_signatures
    )


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
              "metrics": {
                  "ttft_ms": float|None,
                  "latency_ms": float,
                  "model": str,
                  "prompt_tokens": int|None,
                  "completion_tokens": int|None,
                  "total_tokens": int|None,
                  "cost_usd": float|None
              }
            }

    Returns:
        If stream=True: Generator yielding streaming text chunks
        If stream=False: Dict containing response text, usage, and metrics
    """

    client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)

    normalized_messages = (
        [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    )
    
    temperature = 1.0 if _is_o4_model(model) else params.get("temperature")
    
    kwargs = {
        "model": model,
        "messages": normalized_messages,
        "temperature": temperature,
        "top_p": params.get("top_p"),
        "stream": stream,
    }
    
    if _is_o_series_model(model):
        kwargs["max_completion_tokens"] = params.get("max_tokens")
    else:
        kwargs["max_tokens"] = params.get("max_tokens")
    
    if _supports_seed(model):
        kwargs["seed"] = params.get("seed")
    
    response_format = _normalize_response_format(params.get("response_format"))
    if response_format is not None:
        kwargs["response_format"] = response_format

    # Handle tools / tool_choice if provided.
    tools = params.get("tools")
    tool_choice = params.get("tool_choice")

    if tools:
        kwargs["tools"] = [
            t.model_dump() if hasattr(t, "model_dump") else t for t in tools
        ]
    if tool_choice is not None:
        # tool_choice is already a primitive/union type in ChatParams
        kwargs["tool_choice"] = tool_choice

    # If streaming, request usage at end-of-stream
    if stream:
        # if tools:
        #     # For now, we only support tools for non-streaming calls.
        #     raise ValueError(
        #         "Tools/functions are only supported for non-streaming calls in this version."
        #     )
        kwargs["stream_options"] = {"include_usage": True}
    
    t0 = time.perf_counter()

    # For streaming, we perform a single call and then stream chunks as before.
    # For non-streaming, we may perform multiple calls if tool calls are used.
    if stream:
        def _stream_with_tools(current_messages, iteration_count=0, recent_tool_calls=None):
            # Track recent tool calls to detect loops
            if recent_tool_calls is None:
                recent_tool_calls = []
            
            # Prevent infinite loops by limiting iterations
            if iteration_count >= MAX_TOOL_ITERATIONS:
                # Emit error message and stop
                error_msg = f"⚠️ Maximum tool call iterations ({MAX_TOOL_ITERATIONS}) reached. Stopping to prevent infinite loop."
                if stream_mode == "raw":
                    yield "\n" + json.dumps({"error": error_msg}) + "\n"
                else:
                    yield sse_pack(error_msg, event="error")
                return
            
            # Update messages in kwargs
            local_kwargs = kwargs.copy()
            local_kwargs["messages"] = current_messages
            
            # We need to track if we are in a tool call loop
            # If tools are not enabled, this is just a single pass
            
            response_stream = client.chat.completions.create(**local_kwargs)
            
            # Buffer for accumulating tool calls
            # Map index -> {id, name, arguments}
            tool_calls_buffer = {}
            finish_reason = None
            
            # Helper to append chunks to buffer
            for chunk in response_stream:
                # 1. Yield the chunk immediately so text/usage propagates
                yield chunk
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Accumulate tool calls if present
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        
                        if tc.id:
                            tool_calls_buffer[idx]["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[idx]["arguments"] += tc.function.arguments
                
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # End of stream for this turn. Check if we need to execute tools.
            if finish_reason == "tool_calls" and tool_calls_buffer:
                # 1. Reconstruct the assistant message with tool calls
                tool_calls_payload = []
                current_tool_signatures = []  # Track tool name + arguments for loop detection
                for idx in sorted(tool_calls_buffer.keys()):
                    tc_data = tool_calls_buffer[idx]
                    tool_name = tc_data["name"].strip() if tc_data["name"] else ""
                    tool_args = tc_data["arguments"].strip() if tc_data["arguments"] else ""
                    
                    # Skip if tool name is empty or malformed
                    if not tool_name:
                        continue
                    
                    # Detect and fix duplicated tool names (e.g., "get_fmp_company_dataget_fmp_company_data")
                    # This can happen if the name is sent in chunks and gets duplicated
                    name_length = len(tool_name)
                    if name_length > 0:
                        # Check if the name appears to be duplicated (first half == second half)
                        half = name_length // 2
                        if name_length % 2 == 0 and tool_name[:half] == tool_name[half:]:
                            tool_name = tool_name[:half]  # Use only the first half
                    
                    tool_signature = (tool_name, tool_args)
                    current_tool_signatures.append(tool_signature)
                    
                    tool_calls_payload.append({
                        "id": tc_data["id"].strip() if tc_data["id"] else "",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    })
                
                # Check for repeated identical tool calls (loop detection)
                if _is_tool_loop(recent_tool_calls, current_tool_signatures):
                    error_msg = "⚠️ Detected infinite loop: same tool calls repeated. Stopping to prevent infinite loop."
                    if stream_mode == "raw":
                        yield "\n" + json.dumps({"error": error_msg}) + "\n"
                    else:
                        yield sse_pack(error_msg, event="error")
                    return
                
                # Track this tool call set (keep last 5 for loop detection)
                recent_tool_calls.append(current_tool_signatures)
                if len(recent_tool_calls) > 5:
                    recent_tool_calls.pop(0)
                
                # Append assistant message
                current_messages.append({
                    "role": "assistant",
                    "content": None, 
                    "tool_calls": tool_calls_payload
                })
                
                # 2. Execute tools
                executed_tool_names = []
                seen_tool_calls = set()  # Track unique tool calls to avoid duplicates
                for tc in tool_calls_payload:
                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    call_id = tc["id"]
                    
                    # Validate and clean tool name (strip whitespace, handle empty/malformed names)
                    if not func_name or not isinstance(func_name, str):
                        continue
                    func_name = func_name.strip()
                    if not func_name:
                        continue
                    
                    # Create a unique key for this tool call (name + args to handle same tool with different args)
                    tool_call_key = (func_name, func_args_str, call_id)
                    if tool_call_key not in seen_tool_calls:
                        seen_tool_calls.add(tool_call_key)
                        executed_tool_names.append(func_name)
                    
                    result_content = ""
                    try:
                        args = {}
                        if func_args_str and func_args_str.strip():
                            try:
                                args = json.loads(func_args_str)
                            except json.JSONDecodeError:
                                args = {}
                        
                        tool_callable = get_tool_callable(func_name)
                        # Execute
                        if isinstance(args, dict):
                            raw_result = tool_callable(**args)
                        else:
                            raw_result = tool_callable(args)
                        result_content = json.dumps(raw_result)
                    except Exception as exc:
                        result_content = json.dumps({"error": f"Tool execution failed: {exc}"})
                    
                    # Append tool result message
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": result_content
                    })
                
                # 2.5. Emit tool status marker before recursing
                yield ToolStatusMarker(tool_names=executed_tool_names)
                
                # 3. Recurse: Start a new stream with updated messages (increment iteration count, pass recent_tool_calls)
                yield from _stream_with_tools(current_messages, iteration_count=iteration_count + 1, recent_tool_calls=recent_tool_calls)

        res = _stream_with_tools(normalized_messages)
        # res = client.chat.completions.create(**kwargs)

    # --- helpers ------------------------------------------------------------
    def _iter_text(chunks):
        """Yield textual deltas only; ignore non-text chunks. Also passes through ToolStatusMarker."""
        for ch in chunks:
            # Check if this is a ToolStatusMarker
            if isinstance(ch, ToolStatusMarker):
                yield ch, None  # Pass marker with None as the chunk
                continue
                
            piece = None
            # Try delta.content first (standard streaming format)
            try:
                delta = ch.choices[0].delta
                piece = getattr(delta, "content", None)
            except (AttributeError, IndexError, TypeError):
                pass
            
            # Fallback: check message.content (some models like o4-mini may use this for final chunk)
            if not piece:
                try:
                    message = ch.choices[0].message
                    piece = getattr(message, "content", None)
                except (AttributeError, IndexError, TypeError):
                    pass
            
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

    def _update_usage_totals(usage_obj, totals):
        if not usage_obj:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage_obj, key, 0) or 0
            if value:
                totals[key] = (totals[key] or 0) + value

    def _build_metrics_payload(ttft, latency, chunk_count, usage_totals):
        prompt_tokens = usage_totals["prompt_tokens"]
        completion_tokens = usage_totals["completion_tokens"]
        total_tokens = usage_totals["total_tokens"]
        cost_usd = _lookup_pricing(model, prompt_tokens, completion_tokens)
        return {
            "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
            "latency_ms": round(latency * 1000, 1),
            "chunks": chunk_count,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_chunk": (
                round((completion_tokens or 0) / (chunk_count or 1), 3)
                if completion_tokens is not None
                else None
            ),
            "cost_usd": cost_usd,
        }

    # --- streaming path -----------------------------------------------------
    if stream:
        if stream_mode == "raw":
            def _iter_raw() -> Generator[str, None, None]:
                ttft = None
                chunk_count = 0
                usage_totals = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }

                for piece, ch in _iter_text(res):
                    # Handle ToolStatusMarker
                    if isinstance(piece, ToolStatusMarker):
                        for tool_name in _iter_unique_tool_names(piece.tool_names):
                            tool_msg = {"tool_status": f"🔧 `{tool_name}` is executed..."}
                            yield "\n" + json.dumps(tool_msg) + "\n"
                        continue
                    
                    # mark TTFT on first chunk that arrives (text or usage)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    # collect usage if/when the final usage chunk arrives
                    try:
                        _update_usage_totals(getattr(ch, "usage", None), usage_totals)
                    except Exception:
                        pass
                    if piece:
                        chunk_count += 1
                        yield piece

                total = time.perf_counter() - t0
                if include_metrics:
                    metrics = {"metrics": _build_metrics_payload(ttft, total, chunk_count, usage_totals)}
                    # emit a newline + JSON so curl clients can read it cleanly
                    yield "\n" + json.dumps(metrics) + "\n"
            return _iter_raw()

        # SSE mode
        def _iter_sse() -> Generator[bytes, None, None]:
            ttft = None
            chunk_count = 0
            usage_totals = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

            for piece, ch in _iter_text(res):
                # Handle ToolStatusMarker
                if isinstance(piece, ToolStatusMarker):
                    for tool_name in _iter_unique_tool_names(piece.tool_names):
                        tool_msg = f"🔧 `{tool_name}` is executed..."
                        yield sse_pack(tool_msg, event="tool_status")
                    continue
                
                if ttft is None:
                    ttft = time.perf_counter() - t0
                # capture usage if present (arrives as the last/penultimate chunk)
                try:
                    _update_usage_totals(getattr(ch, "usage", None), usage_totals)
                except Exception:
                    pass
                if piece:
                    chunk_count += 1
                    yield sse_pack(piece, event=sse_event)

            total = time.perf_counter() - t0
            if send_done:
                yield sse_pack("done", event="status")
            if include_metrics:
                payload = _build_metrics_payload(ttft, total, chunk_count, usage_totals)
                yield sse_pack(json.dumps(payload), event="metrics")
        return _iter_sse()
        
    # --- non-streaming path (with optional tools) ---------------------------

    def _single_completion_call(current_messages):
        """Helper to perform one non-streaming completion call with given messages."""
        local_kwargs = kwargs.copy()
        local_kwargs["stream"] = False
        local_kwargs["messages"] = current_messages
        # Ensure no leftover stream_options in non-stream calls
        local_kwargs.pop("stream_options", None)
        return client.chat.completions.create(**local_kwargs)

    messages_for_tools = list(normalized_messages)
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    executed_tools = []  # Track which tools were executed
    iteration_count = 0
    recent_tool_calls = []  # Track recent tool calls for loop detection

    while iteration_count < MAX_TOOL_ITERATIONS:
        iteration_count += 1
        res = _single_completion_call(messages_for_tools)
        choice = res.choices[0]
        msg = choice.message

        usage = getattr(res, "usage", None)
        if usage:
            try:
                prompt_tokens_total += getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens_total += getattr(usage, "completion_tokens", 0) or 0
                total_tokens_total += getattr(usage, "total_tokens", 0) or 0
            except Exception:
                pass

        finish_reason = getattr(choice, "finish_reason", None)

        # If the model is requesting tool calls, execute them and loop again.
        if finish_reason == "tool_calls" and getattr(msg, "tool_calls", None):
            # Track current tool call signatures for loop detection
            current_tool_signatures = []
            for tc in msg.tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                arguments_str = getattr(fn, "arguments", "{}") if fn else "{}"
                current_tool_signatures.append((name, arguments_str))
            
            # Check for repeated identical tool calls (loop detection)
            if _is_tool_loop(recent_tool_calls, current_tool_signatures):
                final_content = "⚠️ Detected infinite loop: same tool calls repeated. Stopping to prevent infinite loop."
                break
            
            # Track this tool call set (keep last 5 for loop detection)
            recent_tool_calls.append(current_tool_signatures)
            if len(recent_tool_calls) > 5:
                recent_tool_calls.pop(0)
            # Record the assistant message that requested tool calls
            tool_calls_payload = []
            tool_result_messages = []
            for tc in msg.tool_calls:
                tc_id = getattr(tc, "id", None)
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                arguments_str = getattr(fn, "arguments", "{}") if fn else "{}"

                tool_calls_payload.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_str,
                        },
                    }
                )

                # Execute the corresponding Python tool
                if name:
                    # Track that this tool was executed
                    if name not in executed_tools:
                        executed_tools.append(name)
                    try:
                        args = {}
                        if isinstance(arguments_str, str) and arguments_str.strip():
                            try:
                                args = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                # Fall back to empty args if parsing fails
                                args = {}
                        tool_callable = get_tool_callable(name)
                        result = tool_callable(**args) if isinstance(args, dict) else tool_callable(args)
                    except Exception as exc:
                        # Surface tool execution errors back to the model
                        result = {"error": f"Tool '{name}' failed: {exc}"}

                    # Collect the tool result message; we'll append after the assistant message.
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

            # Append the assistant message that contained the tool_calls first,
            # then the tool result messages, to preserve chronological order.
            messages_for_tools.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_payload,
                }
            )
            messages_for_tools.extend(tool_result_messages)

            # Loop again with updated messages to let the model use tool results.
            continue
        
        # If we reach here and iteration_count >= MAX_TOOL_ITERATIONS, break to prevent infinite loop
        if iteration_count >= MAX_TOOL_ITERATIONS:
            final_content = msg.content or ""
            if not final_content:
                final_content = f"⚠️ Maximum tool call iterations ({MAX_TOOL_ITERATIONS}) reached. Stopping to prevent infinite loop."
            break

        # Otherwise, we have a normal completion (no further tool_calls).
        final_content = msg.content or ""
        total = time.perf_counter() - t0
        cost_usd = _lookup_pricing(model, prompt_tokens_total, completion_tokens_total)

        # Format tool execution messages (same format as streaming)
        tool_messages = []
        if executed_tools:
            for tool_name in executed_tools:
                tool_messages.append(f"🔧 `{tool_name}` is executed...")

        return {
            "text": final_content,
            "metrics": {
                "ttft_ms": None,  # not meaningful for non-streaming
                "latency_ms": round(total * 1000, 1),
                "model": model,
                "prompt_tokens": prompt_tokens_total or None,
                "completion_tokens": completion_tokens_total or None,
                "total_tokens": total_tokens_total or None,
                "cost_usd": cost_usd,
                "executed_tools": executed_tools if executed_tools else None,
            },
            "tool_messages": tool_messages,  # Formatted messages for UI display
        }