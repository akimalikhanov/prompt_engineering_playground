# app_gradio.py
import gradio as gr
from utils.sse import sse_data
import time


def chat_fn(user_test: str):
    url = "http://localhost:8000/chat.streamsse"
    payload={
        "provider_id": "google",
        "model_id": "gemini-flash-lite",
        "messages": [{"role": "user", "content": f"{user_test}"}],
        "params": {}
    }

    acc = ""
    for ev, data in sse_data(url, payload, method="POST", timeout=None):
        if ev == "metrics":
            print("\nMETRICS:", data) # will log later
            continue
        if data in ("[DONE]", "done"):
            break
        acc += data
        # print(acc, end="\r", flush=True)
        yield acc

demo = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(label="Your prompt"),
    outputs=gr.Textbox(label="Assistant (streaming)", lines=10, show_copy_button=True),
    title="SSE Streaming Demo"
)

if __name__ == "__main__":
    demo.queue().launch()

