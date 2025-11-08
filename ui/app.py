import json
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx

from config.load_models_config import _load_models_config


@lru_cache(maxsize=1)
def _load_models() -> Dict[str, Any]:
    config = _load_models_config()
    models: List[Dict[str, Any]] = []
    for model_cfg in config.get("models", []):
        models.append(
            {
                "id": model_cfg["id"],
                "label": model_cfg["label"],
                "provider": model_cfg["provider"],
            }
        )
    defaults = config.get("defaults", {}).get("params", {})
    return {
        "models": models,
        "defaults": {
            "temperature": defaults.get("temperature", 0.7),
            "top_p": defaults.get("top_p", 1.0),
            "max_tokens": defaults.get("max_tokens", 512),
            "seed": defaults.get("seed"),
        },
    }


MODELS_DATA = _load_models()
MODEL_LOOKUP = {model["id"]: model for model in MODELS_DATA["models"]}
DEFAULT_MODEL_ID = "gemini-flash-lite" if "gemini-flash-lite" in MODEL_LOOKUP else next(iter(MODEL_LOOKUP.keys()), None)

if DEFAULT_MODEL_ID is None:
    raise RuntimeError("No models configured in config/models.yaml")

DEFAULT_API_BASE_URL =os.getenv("API_BASE_URL", "http://localhost:8000")

ENDPOINTS = {
    "chat": {
        "label": "Non-streaming JSON (/chat)",
        "path": "/chat",
        "stream": False,
        "sse": False,
    },
    "chat.stream": {
        "label": "Streaming text (/chat.stream)",
        "path": "/chat.stream",
        "stream": True,
        "sse": False,
    },
    "chat.streamsse": {
        "label": "Streaming SSE (/chat.streamsse)",
        "path": "/chat.streamsse",
        "stream": True,
        "sse": True,
    },
}

DEFAULT_ENDPOINT_KEY = "chat.streamsse" if "chat.streamsse" in ENDPOINTS else next(iter(ENDPOINTS.keys()))

BADGES_CSS = """
.badges {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 0.5rem;
}
.badge {
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    border: 1px solid rgba(15, 23, 42, 0.16);
}
.badge-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
}
.badge-value {
    font-weight: 600;
}
@media (prefers-color-scheme: dark) {
    .badge {
        border-color: rgba(148, 163, 184, 0.35);
    }
    .badge-label {
    }
    .badge-value {
    }
}
"""


def _render_badge(label: str, value: Optional[str]) -> str:
    display = value if value is not None else "—"
    return f"<span class='badge'><span class='badge-label'>{label}</span><span class='badge-value'>{display}</span></span>"


def _format_tokens(metrics: Dict[str, Any]) -> Optional[str]:
    prompt = metrics.get("prompt_tokens")
    completion = metrics.get("completion_tokens")
    total = metrics.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    parts = []
    if prompt is not None or completion is not None:
        prompt_val = prompt if prompt is not None else "?"
        completion_val = completion if completion is not None else "?"
        parts.append(f"{prompt_val}/{completion_val}")
    if total is not None:
        parts.append(f"({total})")
    return " ".join(parts) if parts else str(total)


def _format_cost(metrics: Dict[str, Any]) -> Optional[str]:
    cost = metrics.get("cost_usd")
    if cost is None:
        return None
    return f"${cost:.4f}"


def format_metrics_badges(metrics: Optional[Dict[str, Any]], ttft_fallback: Optional[float], latency_fallback: Optional[float]) -> str:
    metrics = metrics or {}
    ttft_ms = metrics.get("ttft_ms") or ttft_fallback
    latency_ms = metrics.get("latency_ms") or latency_fallback

    badges = [
        _render_badge("TTFT", f"{ttft_ms:.0f} ms" if ttft_ms is not None else None),
        _render_badge("Latency", f"{latency_ms:.0f} ms" if latency_ms is not None else None),
        _render_badge("Tokens", _format_tokens(metrics)),
        _render_badge("Cost", _format_cost(metrics)),
    ]
    return f"<div class='badges'>{''.join(badges)}</div>"


def _build_messages(
    user_message: str,
    history: List[Tuple[str, str]],
    system_prompt: Optional[str],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for past_user, past_assistant in history:
        if past_user:
            messages.append({"role": "user", "content": past_user})
        if past_assistant:
            messages.append({"role": "assistant", "content": past_assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def _coerce_seed(seed_text: Optional[str]) -> Optional[int]:
    if seed_text is None:
        return None
    if isinstance(seed_text, (int, float)):
        try:
            seed_val = int(seed_text)
        except ValueError:
            return None
        return max(0, seed_val)
    seed_text = seed_text.strip()
    if not seed_text:
        return None
    try:
        seed_val = int(seed_text)
    except ValueError:
        return None
    return max(0, seed_val)


def _normalize_selection(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("value") or value.get("id")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[1]
    return None


def stream_chat(
    user_message: str,
    history: List[Tuple[str, str]],
    model_choice: Any,
    endpoint_choice: Any,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed_input: Optional[str],
    system_prompt: Optional[str],
    api_base_url: str,
) -> Tuple[List[Tuple[str, str]], str]:
    history = history or []

    if not user_message or not user_message.strip():
        return history, format_metrics_badges({}, None, None)

    normalized_choice = _normalize_selection(model_choice)
    model_config = MODEL_LOOKUP.get(normalized_choice) if normalized_choice else None
    if not model_config:
        history = history + [(user_message, "⚠️ Unknown model selection.")]
        return history, format_metrics_badges({}, None, None)

    endpoint_key = _normalize_selection(endpoint_choice)
    endpoint_cfg = ENDPOINTS.get(endpoint_key) if endpoint_key else None
    if not endpoint_cfg:
        history = history + [(user_message, "⚠️ Unknown endpoint selection.")]
        return history, format_metrics_badges({}, None, None)

    provider_id = model_config["provider"]
    payload_messages = _build_messages(user_message, history, system_prompt)

    api_url = (api_base_url or DEFAULT_API_BASE_URL).strip()
    if not api_url:
        api_url = DEFAULT_API_BASE_URL
    endpoint_url = f"{api_url.rstrip('/')}{endpoint_cfg['path']}"

    params: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
    }

    seed_value = _coerce_seed(seed_input)
    if seed_value is not None:
        params["seed"] = seed_value

    request_body = {
        "provider_id": provider_id,
        "model_id": model_config["id"],
        "messages": payload_messages,
        "params": params,
    }

    running_history = history + [(user_message, "")]
    ttft_ms: Optional[float] = None
    metrics: Dict[str, Any] = {}
    start_time = time.perf_counter()

    # Emit initial state to display user message immediately.
    yield running_history, format_metrics_badges(metrics, ttft_ms, None)

    accumulated_response = ""
    timeout = httpx.Timeout(60.0, connect=10.0, read=None, write=10.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            if endpoint_cfg["stream"]:
                with client.stream("POST", endpoint_url, json=request_body) as response:
                    response.raise_for_status()
                    if endpoint_cfg["sse"]:
                        done = False
                        sse_buffer = ""
                        for chunk in response.iter_bytes():
                            if done:
                                break
                            if not chunk:
                                continue
                            sse_buffer += chunk.decode("utf-8")
                            while "\n\n" in sse_buffer and not done:
                                event, sse_buffer = sse_buffer.split("\n\n", 1)
                                event_str = event.strip()
                                if not event_str:
                                    continue
                                lowered = event_str.lower()
                                if "event: done" in lowered or "data: [done]" in lowered:
                                    done = True
                                    break
                                if "event: metrics" in event_str:
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            try:
                                                metrics_data = json.loads(line[6:])
                                                metrics.clear()
                                                metrics.update(metrics_data)
                                            except json.JSONDecodeError:
                                                pass
                                    continue
                                if "event: message" in event_str:
                                    data_lines = []
                                    for line in event_str.split("\n"):
                                        if line.startswith("data: "):
                                            data_lines.append(line[6:])
                                    new_text = "".join(data_lines)
                                    if not new_text:
                                        continue
                                    if ttft_ms is None:
                                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                    accumulated_response += new_text
                                    running_history[-1] = (user_message, accumulated_response)
                                    latency_now = (time.perf_counter() - start_time) * 1000.0
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now)
                        if not done and sse_buffer.strip():
                            event_str = sse_buffer.strip()
                            if "event: metrics" in event_str:
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        try:
                                            metrics_data = json.loads(line[6:])
                                            metrics.clear()
                                            metrics.update(metrics_data)
                                        except json.JSONDecodeError:
                                            pass
                    else:
                        for chunk in response.iter_text():
                            if not chunk:
                                continue
                            if chunk.startswith("\n{") and "metrics" in chunk:
                                try:
                                    metrics_obj = json.loads(chunk.strip())
                                    metrics = metrics_obj.get("metrics", metrics_obj)
                                except json.JSONDecodeError:
                                    pass
                                continue

                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000.0

                            accumulated_response += chunk
                            running_history[-1] = (user_message, accumulated_response)
                            latency_now = (time.perf_counter() - start_time) * 1000.0
                            yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now)
            else:
                response = client.post(endpoint_url, json=request_body)
                response.raise_for_status()
                data = response.json()
                text = data.get("text", "")
                metrics = data.get("metrics", {})

                if text:
                    accumulated_response = text
                    running_history[-1] = (user_message, accumulated_response)
                else:
                    running_history[-1] = (user_message, "⚠️ Empty response body.")

                if ttft_ms is None and metrics.get("ttft_ms") is not None:
                    ttft_ms = metrics["ttft_ms"]
                final_latency = (time.perf_counter() - start_time) * 1000.0
                yield running_history, format_metrics_badges(metrics, ttft_ms, final_latency)
                return

    except httpx.HTTPStatusError as http_err:
        response = http_err.response
        detail_text = ""
        try:
            content_bytes = response.read()
            if content_bytes:
                detail_text = content_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            detail_text = ""
        if not detail_text:
            detail_text = getattr(response, "reason_phrase", "") or str(http_err)
        error_msg = f"⚠️ API error {response.status_code}: {detail_text}"
        running_history[-1] = (user_message, error_msg)
        yield running_history, format_metrics_badges(metrics, ttft_ms, None)
        return
    except httpx.HTTPError as transport_err:
        error_msg = f"⚠️ Connection error: {transport_err}"
        running_history[-1] = (user_message, error_msg)
        yield running_history, format_metrics_badges(metrics, ttft_ms, None)
        return
    except Exception as exc:
        running_history[-1] = (user_message, f"⚠️ Unexpected error: {exc}")
        yield running_history, format_metrics_badges(metrics, ttft_ms, None)
        return

    final_latency = (time.perf_counter() - start_time) * 1000.0
    running_history[-1] = (user_message, accumulated_response)
    yield running_history, format_metrics_badges(metrics, ttft_ms, final_latency)


def build_demo() -> gr.Blocks:
    models_choices = [
        (
            f"{model['label']} ({model['provider']}/{model['id']})",
            model["id"],
        )
        for model in MODELS_DATA["models"]
    ]
    endpoint_choices = [
        (
            endpoint_cfg["label"],
            endpoint_key,
        )
        for endpoint_key, endpoint_cfg in ENDPOINTS.items()
    ]
    defaults = MODELS_DATA["defaults"]

    with gr.Blocks(css=BADGES_CSS, theme=gr.themes.Ocean(primary_hue="emerald"), title="Prompt Engineering Playground") as demo:
        gr.Markdown(
            "# Prompt Engineering Playground\n"
            "Interact with the various LLM models via a lightweight Gradio UI."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    bubble_full_width=False,
                )
                user_input = gr.Textbox(
                    label="Your message (Enter for new line, Shift+Enter to send)",
                    placeholder="Ask something...",
                    autofocus=True,
                    lines=3,
                    elem_id="prompt-input",
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary", elem_id="send-button")
                    clear_button = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=1, min_width=280):
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=models_choices,
                    value=DEFAULT_MODEL_ID,
                )
                endpoint_dropdown = gr.Dropdown(
                    label="Endpoint",
                    choices=endpoint_choices,
                    value=DEFAULT_ENDPOINT_KEY,
                )
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=defaults.get("temperature", 0.7),
                )
                top_p_slider = gr.Slider(
                    label="Top-p",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=defaults.get("top_p", 1.0),
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=1,
                    maximum=4000,
                    step=1,
                    value=defaults.get("max_tokens", 512),
                )
                seed_text = gr.Textbox(
                    label="Seed (optional)",
                    placeholder="e.g. 42",
                )
                system_prompt_box = gr.Textbox(
                    label="System prompt (optional)",
                    placeholder="Add a system instruction to steer responses.",
                    lines=3,
                )
                api_base_box = gr.Textbox(
                    label="API base URL",
                    value=DEFAULT_API_BASE_URL,
                    placeholder="http://localhost:8000",
                )
                metrics_display = gr.Markdown(
                    value=format_metrics_badges({}, None, None),
                    label="Metrics",
                )

        submit_inputs = [
            user_input,
            chatbot,
            model_dropdown,
            endpoint_dropdown,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            seed_text,
            system_prompt_box,
            api_base_box,
        ]

        submit_outputs = [chatbot, metrics_display]

        user_input.submit(stream_chat, inputs=submit_inputs, outputs=submit_outputs, queue=True)
        user_input.submit(lambda: "", inputs=None, outputs=user_input, queue=False)

        def submit_chat(*args):
            for value in stream_chat(*args):
                yield value
        send_button.click(submit_chat, inputs=submit_inputs, outputs=submit_outputs, queue=True)
        send_button.click(lambda: "", inputs=None, outputs=user_input, queue=False)

        clear_button.click(
            lambda: ([], format_metrics_badges({}, None, None)),
            inputs=None,
            outputs=[chatbot, metrics_display],
            queue=False,
        )

        demo.queue()

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()

