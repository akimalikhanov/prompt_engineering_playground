from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Generator
from services.errors import BackendError
from config.load_models_config import _load_models_config
from schemas.schemas import *
from services.router import route_call

_models_config = _load_models_config()
_models_list = [model['label'] for model in _models_config['models']]

app = FastAPI(title="Prompt Engineering Playground")

# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(status="ok", message="Hello from FastAPI")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=_models_list)

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={200: {"description": "Complete response as JSON"}},
)
def chat(body: ChatRequest) -> ChatResponse:
    result = route_call(
        provider_id=body.provider_id,
        model_id=body.model_id,
        messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
        params=body.params.model_dump(),
        stream=False,
    )
    return ChatResponse(
        content=result,
        model_id=body.model_id,
        provider_id=body.provider_id,
    )

@app.post(
    "/chat.stream",
    response_model=None,  # streaming is not a Pydantic field
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Streaming of raw text chunks (inline; clients append to the same line)",
        }
    },
)
def chat_stream(body: ChatRequest):
    gen = route_call(
        provider_id=body.provider_id,
        model_id=body.model_id,
        messages=[m.model_dump() for m in body.messages] if isinstance(body.messages, list) else body.messages,
        params=body.params.model_dump(),
        stream=True,
    )

    def inline_wrap() -> Generator[bytes, None, None]:
        try:
            for chunk in gen:
                if not chunk:
                    continue
                # Stream raw chunk bytes; clients should render as a running line
                yield chunk.encode("utf-8")
        except Exception as e:
            yield f"[error] {str(e)}\n".encode("utf-8")

    return StreamingResponse(inline_wrap(), media_type="text/plain")


@app.exception_handler(BackendError)
def handle_backend_error(request: Request, exc: BackendError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )
