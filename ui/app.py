import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import httpx

from utils.load_configs import (
    _load_models,
    _load_models_config,
    _load_gradio_templates,
    _load_endpoints,
    _get_default_endpoint_key,
    _load_delimiter_definitions,
    _load_response_schema_templates,
)
from ui.styles import BADGES_CSS
from utils.ui_formatters import format_metrics_badges
from utils.message_utils import _build_messages, _coerce_seed, _build_response_format, _normalize_selection
from utils.delimiter_utils import RESOLVER_FUNCTIONS, _delim_pair, _insert_pair_at_end, _wrap_entire_message
from utils.model_params_utils import get_effective_params_for_model, _get_model_special_behavior, _get_param_values_for_model


MODELS_DATA = _load_models()
MODEL_LOOKUP = {model["id"]: model for model in MODELS_DATA["models"]}
DEFAULT_MODEL_ID = "gemini-flash-lite" if "gemini-flash-lite" in MODEL_LOOKUP else next(iter(MODEL_LOOKUP.keys()), None)

if DEFAULT_MODEL_ID is None:
    raise RuntimeError("No models configured in config/models.yaml")

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

ENDPOINTS = _load_endpoints()
DEFAULT_ENDPOINT_KEY = _get_default_endpoint_key()

# Load delimiter and schema template configurations
DELIMITER_DEFINITIONS: Dict[str, Dict[str, Any]] = _load_delimiter_definitions()
DELIMITER_CHOICES: List[str] = list(DELIMITER_DEFINITIONS.keys())

RESPONSE_SCHEMA_TEMPLATES: Dict[str, Dict[str, Any]] = _load_response_schema_templates()
RESPONSE_SCHEMA_CHOICES: List[str] = list(RESPONSE_SCHEMA_TEMPLATES.keys())


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
    context_prompt: Optional[str],
    api_base_url: str,
    response_mode: str,
    strict_schema: bool,
    schema_code: Optional[str],
    schema_template: Optional[str] = None,
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    """Stream chat responses. Always yields values, even for errors."""
    history = history or []

    if not user_message or not user_message.strip():
        yield history, format_metrics_badges({}, None, None)
        return

    # Validate response_format early and show error in UI if needed
    if response_mode == "JSON schema":
        if not schema_code or not schema_code.strip():
            error_msg = "⚠️ JSON schema mode requires a schema. Please enter a valid JSON schema."
            history = history + [(user_message, error_msg)]
            yield history, format_metrics_badges({}, None, None)
            return
        try:
            # Validate JSON before proceeding
            parsed_schema = json.loads(schema_code.strip())
            # Also validate it's a dict
            if not isinstance(parsed_schema, dict):
                error_msg = f"⚠️ JSON schema must be an object/dict, got {type(parsed_schema).__name__}."
                history = history + [(user_message, error_msg)]
                yield history, format_metrics_badges({}, None, None)
                return
        except json.JSONDecodeError as e:
            error_msg = f"⚠️ Invalid JSON schema: {str(e)}\n\nPlease check your JSON syntax (missing brackets, commas, etc.)."
            history = history + [(user_message, error_msg)]
            yield history, format_metrics_badges({}, None, None)
            return

    normalized_choice = _normalize_selection(model_choice)
    model_config = MODEL_LOOKUP.get(normalized_choice) if normalized_choice else None
    if not model_config:
        error_msg = "⚠️ Unknown model selection."
        history = history + [(user_message, error_msg)]
        yield history, format_metrics_badges({}, None, None)
        return

    endpoint_key = _normalize_selection(endpoint_choice)
    endpoint_cfg = ENDPOINTS.get(endpoint_key) if endpoint_key else None
    if not endpoint_cfg:
        error_msg = "⚠️ Unknown endpoint selection."
        history = history + [(user_message, error_msg)]
        yield history, format_metrics_badges({}, None, None)
        return

    provider_id = model_config["provider"]
    payload_messages = _build_messages(user_message, history, system_prompt, context_prompt)

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

    # Handle response_format
    schema_template_label = _normalize_selection(schema_template) if schema_template else None
    response_format = _build_response_format(response_mode, strict_schema, schema_code, schema_template_label, RESPONSE_SCHEMA_TEMPLATES)
    if response_format is not None:
        params["response_format"] = response_format
        # Debug: log what we're sending
        print(f"DEBUG: Adding response_format to params: {json.dumps(response_format, indent=2)}")
    else:
        # Debug: log why response_format is None
        print(f"DEBUG: response_format is None (mode={response_mode}, schema_code={'present' if schema_code else 'None'})")

    request_body = {
        "provider_id": provider_id,
        "model_id": model_config["id"],
        "messages": payload_messages,
        "params": params,
        "context_prompt": context_prompt.strip() if context_prompt and context_prompt.strip() else None,
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
                        # Process any remaining buffer content (incomplete final event)
                        if not done and sse_buffer.strip():
                            event_str = sse_buffer.strip()
                            # Process metrics events
                            if "event: metrics" in event_str:
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        try:
                                            metrics_data = json.loads(line[6:])
                                            metrics.clear()
                                            metrics.update(metrics_data)
                                        except json.JSONDecodeError:
                                            pass
                            # Also process message events in the final buffer
                            elif "event: message" in event_str:
                                data_lines = []
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        data_lines.append(line[6:])
                                new_text = "".join(data_lines)
                                if new_text:
                                    if ttft_ms is None:
                                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                    accumulated_response += new_text
                                    running_history[-1] = (user_message, accumulated_response)
                                    latency_now = (time.perf_counter() - start_time) * 1000.0
                                    yield running_history, format_metrics_badges(metrics, ttft_ms, latency_now)
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




# --- Helpers (moved out of build_demo) ---------------------------------------

def sync_delimiter_inputs(selected_style: str):
    definition = DELIMITER_DEFINITIONS.get(selected_style or "")
    if not definition:
        return gr.update(visible=False), gr.update(visible=False)

    start_update = gr.update(
        visible=definition.get("requires_start", False),
        label=definition.get("start_label", "Custom start"),
        placeholder=definition.get("start_placeholder", "e.g. [[[INSTRUCT]]]"),
    )
    end_update = gr.update(
        visible=definition.get("requires_end", False),
        label=definition.get("end_label", "Custom end"),
        placeholder=definition.get("end_placeholder", "e.g. [[[/INSTRUCT]]]"),
    )
    return start_update, end_update


def handle_insert_end(text, style, cs, ce):
    return _insert_pair_at_end(text, style, cs, ce, DELIMITER_DEFINITIONS)


def handle_wrap_all(text, style, cs, ce):
    return _wrap_entire_message(text, style, cs, ce, DELIMITER_DEFINITIONS)


def clear_chat_and_metrics():
    return [], format_metrics_badges({}, None, None)


def on_schema_template_change(template_label: str, current_strict: bool) -> Tuple[str, bool]:
    """Handle schema template selection.
    
    When a template is selected, convert its schema to JSON and update the code editor.
    For "Custom", return empty template.
    
    Args:
        template_label: Selected template label from dropdown
        current_strict: Current value of strict checkbox
        
    Returns:
        Tuple of (schema_json_string, strict_checkbox_value)
    """
    if not template_label or template_label == "Custom...":
        # Return empty template for custom
        default_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": []
        }
        return json.dumps(default_schema, indent=2), current_strict
    
    template = RESPONSE_SCHEMA_TEMPLATES.get(template_label)
    if not template:
        # Fallback if template not found
        default_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": []
        }
        return json.dumps(default_schema, indent=2), current_strict
    
    # Extract schema from template (it's already a dict from YAML)
    json_schema_config = template.get("json_schema", {})
    schema_dict = json_schema_config.get("schema", {})
    
    # Convert dict to JSON string with indentation
    schema_json = json.dumps(schema_dict, indent=2)
    
    # Keep current strict value unchanged
    return schema_json, current_strict


def reset_textbox():
    return ""


def update_params_for_model(model_choice: Any):
    """Update all parameter UI components based on selected model.
    
    This is a generic function that:
    1. Looks up the model's params_override from models.yaml
    2. Merges with defaults to get effective params
    3. Applies special behavior rules (e.g., fixed values, disabled UI)
    4. Returns updates for all parameter UI components
    
    Returns:
        Tuple of gr.update() calls for all parameter UI components in order:
        (temperature_slider, top_p_slider, max_tokens_slider, seed_textbox)
    """
    normalized_choice = _normalize_selection(model_choice)
    params = _get_param_values_for_model(normalized_choice, MODELS_DATA, MODEL_LOOKUP)
    
    return (
        gr.update(**params["temperature"]),
        gr.update(value=params["top_p"]["value"]),
        gr.update(value=params["max_tokens"]["value"]),
        gr.update(value=params["seed"]["value"]),
    )


def submit_chat_generator(*args):
    for value in stream_chat(*args):
        yield value


# --- Main UI builder ---------------------------------------------------------

def build_demo() -> gr.Blocks:
    models_choices = [
        (
            f"{model['label']} ({model['provider']})",
            model["id"],
        )
        for model in MODELS_DATA["models"]
    ]
    endpoint_choices = [
        (endpoint_cfg["label"], endpoint_key)
        for endpoint_key, endpoint_cfg in ENDPOINTS.items()
    ]
    defaults = MODELS_DATA["defaults"]

    with gr.Blocks(
        css=BADGES_CSS,
        theme=gr.themes.Ocean(primary_hue="emerald"),
        title="Prompt Engineering Playground",
    ) as demo:
        gr.Markdown(
            "# Prompt Engineering Playground\n"
            "Interact with various LLM models via a lightweight Gradio UI."
        )

        with gr.Row():
            # --- Left: chat area ------------------------------------------------
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    bubble_full_width=False,
                )
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask something...",
                    autofocus=True,
                    lines=3,
                    elem_id="prompt-input",
                )
                with gr.Row():
                    send_button = gr.Button(
                        "Send", variant="primary", elem_id="send-button"
                    )
                    clear_button = gr.Button("Clear", variant="secondary")

                metrics_display = gr.Markdown(
                    value=format_metrics_badges({}, None, None),
                    label="Metrics",
                )

            # --- Right: settings / parameters ----------------------------------
            with gr.Column(scale=2, min_width=420):
                with gr.Accordion("Parameters", open=True):
                    # General
                    with gr.Row():
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

                    with gr.Accordion("API settings", open=False):
                        api_base_box = gr.Textbox(
                            label="API base URL",
                            value=DEFAULT_API_BASE_URL,
                            placeholder="http://localhost:8000",
                        )

                    # Tabs for logical grouping
                    with gr.Tabs():
                        # --- Generation tab ------------------------------------
                        with gr.Tab("Generation"):
                            gr.Markdown(
                                "Control sampling behavior. Lower temperature/top-p "
                                "for more deterministic outputs; higher for more creative."
                            )
                            with gr.Row():
                                # Initialize parameters based on default model (merge defaults with params_override)
                                initial_params = _get_param_values_for_model(DEFAULT_MODEL_ID, MODELS_DATA, MODEL_LOOKUP)
                                
                                temperature_slider = gr.Slider(
                                    label=initial_params["temperature"]["label"],
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1,
                                    value=initial_params["temperature"]["value"],
                                    interactive=initial_params["temperature"]["interactive"],
                                )
                                
                                top_p_slider = gr.Slider(
                                    label="Top-p",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=initial_params["top_p"]["value"],
                                )
                            with gr.Row():
                                max_tokens_slider = gr.Slider(
                                    label="Max tokens",
                                    minimum=1,
                                    maximum=4000,
                                    step=1,
                                    value=initial_params["max_tokens"]["value"],
                                )
                                
                                seed_text = gr.Textbox(
                                    label="Seed (optional)",
                                    placeholder="e.g. 42",
                                    value=initial_params["seed"]["value"],
                                )

                            # --- Structured output (response_format) ------------
                            with gr.Accordion("Structured output (optional)", open=False):
                                gr.Markdown(
                                    "Use `response_format` to force JSON output.\n\n"
                                    "- **None**: normal text response\n"
                                    "- **JSON object**: any valid JSON object\n"
                                    "- **JSON schema**: enforce a specific JSON structure"
                                )

                                response_mode_dd = gr.Dropdown(
                                    label="Mode",
                                    choices=["None", "JSON object", "JSON schema"],
                                    value="None",
                                )

                                strict_ckb = gr.Checkbox(
                                    label="Strict schema (JSON only, no extra fields)",
                                    value=True,
                                    visible=False,
                                )

                                schema_template_dd = gr.Dropdown(
                                    label="Schema template",
                                    choices=RESPONSE_SCHEMA_CHOICES,
                                    value="Custom...",
                                    visible=False,
                                    interactive=True,
                                )

                                schema_code = gr.Code(
                                    label="JSON schema",
                                    language="json",
                                    value='{\n  "type": "object",\n  "additionalProperties": false,\n'
                                          '  "properties": {},\n  "required": []\n}',
                                    visible=False,
                                    lines=12,
                                    max_lines=20,
                                )

                                # Small change handler to show/hide fields when mode changes
                                def on_response_mode_change(mode):
                                    show_schema = mode == "JSON schema"
                                    return (
                                        gr.update(visible=show_schema),  # strict_ckb
                                        gr.update(visible=show_schema),  # schema_template_dd
                                        gr.update(visible=show_schema),  # schema_code
                                    )

                                response_mode_dd.change(
                                    on_response_mode_change,
                                    inputs=[response_mode_dd],
                                    outputs=[strict_ckb, schema_template_dd, schema_code],
                                    queue=False,
                                )

                                # Handler for schema template selection
                                def on_schema_template_select(template_label: str, current_strict: bool):
                                    """Handle template selection and update schema code and strict checkbox."""
                                    schema_json, strict_val = on_schema_template_change(template_label, current_strict)
                                    return (
                                        gr.update(value=schema_json),  # schema_code
                                        gr.update(value=strict_val),  # strict_ckb
                                    )

                                schema_template_dd.change(
                                    on_schema_template_select,
                                    inputs=[schema_template_dd, strict_ckb],
                                    outputs=[schema_code, strict_ckb],
                                    queue=False,
                                )

                        # --- Prompting tab -------------------------------------
                        with gr.Tab("Prompting"):
                            gr.Markdown(
                                "Use system prompt for **high-level behavior**, "
                                "and Context to **ground answers in facts**."
                            )
                            system_prompt_box = gr.Textbox(
                                label="System prompt (optional)",
                                placeholder="Add global instructions, tone, or role.",
                                lines=3,
                            )

                            context_box = gr.Textbox(
                                label="Context (optional)",
                                placeholder=(
                                    "Add retrieved facts or reference material here. "
                                    "This will be sent as a special `CONTEXT:` user "
                                    "message before your main question."
                                ),
                                lines=3,
                            )

                            with gr.Accordion("Delimiters (advanced)", open=False):
                                gr.Markdown(
                                    "Delimiters create clear **boundaries** in prompts "
                                    "between instructions, context, and answers. "
                                    "They help parsing and reduce prompt injection risks."
                                )

                                delim_pair_dd = gr.Dropdown(
                                    choices=DELIMITER_CHOICES,
                                    value=DELIMITER_CHOICES[0],
                                    label="Delimiter style",
                                    interactive=True,
                                )
                                with gr.Row():
                                    custom_start_tb = gr.Textbox(
                                        label="Custom start",
                                        placeholder="e.g. [[[INSTRUCT]]]",
                                        visible=False,
                                    )
                                    custom_end_tb = gr.Textbox(
                                        label="Custom end",
                                        placeholder="e.g. [[[/INSTRUCT]]]",
                                        visible=False,
                                    )

                                delim_pair_dd.change(
                                    sync_delimiter_inputs,
                                    inputs=[delim_pair_dd],
                                    outputs=[custom_start_tb, custom_end_tb],
                                    queue=False,
                                )

                                with gr.Row():
                                    insert_delim_btn = gr.Button(
                                        "Insert at end", variant="secondary"
                                    )
                                    wrap_all_btn = gr.Button(
                                        "Wrap all text", variant="secondary"
                                    )

                                insert_delim_btn.click(
                                    handle_insert_end,
                                    inputs=[
                                        user_input,
                                        delim_pair_dd,
                                        custom_start_tb,
                                        custom_end_tb,
                                    ],
                                    outputs=[user_input],
                                    queue=False,
                                )
                                wrap_all_btn.click(
                                    handle_wrap_all,
                                    inputs=[
                                        user_input,
                                        delim_pair_dd,
                                        custom_start_tb,
                                        custom_end_tb,
                                    ],
                                    outputs=[user_input],
                                    queue=False,
                                )

                        # --- (Optional) More tabs later: Tools / Functions / RAG etc.
                        # with gr.Tab("Advanced"):
                        #     ...

        # --- Wiring submissions -----------------------------------------------
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
            context_box,
            api_base_box,
            response_mode_dd,
            strict_ckb,
            schema_code,
            schema_template_dd,
        ]
        submit_outputs = [chatbot, metrics_display]

        user_input.submit(stream_chat, inputs=submit_inputs, outputs=submit_outputs, queue=True)
        user_input.submit(reset_textbox, inputs=None, outputs=user_input, queue=False)

        send_button.click(
            submit_chat_generator,
            inputs=submit_inputs,
            outputs=submit_outputs,
            queue=True,
        )
        send_button.click(reset_textbox, inputs=None, outputs=user_input, queue=False)

        clear_button.click(
            clear_chat_and_metrics,
            inputs=None,
            outputs=[chatbot, metrics_display],
            queue=False,
        )

        # Update all parameter UI components when model changes
        model_dropdown.change(
            update_params_for_model,
            inputs=[model_dropdown],
            outputs=[temperature_slider, top_p_slider, max_tokens_slider, seed_text],
            queue=False,
        )

        demo.queue()

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

