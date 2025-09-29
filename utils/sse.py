from typing import Tuple, Optional, Iterator
import requests
import json
import httpx


def sse_pack(data: str, *, event: Optional[str] = None,
             id: Optional[str] = None, retry: Optional[int] = None) -> bytes:
    """
    Produce a single SSE message frame:
      id: <id>\n
      event: <event>\n
      retry: <ms>\n
      data: <line>\n  (repeat for multiline)
      \n
    """
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")  # blank line to terminate the message
    return ("\n".join(lines) + "\n").encode("utf-8")
    
def sse_data(
    url: str,
    payload: Optional[dict] = None,
    method: str = "POST",
    timeout: Optional[float] = None,
) -> Iterator[Tuple[Optional[str], str]]:
    """
    Minimal SSE client: yields (event_name, data_text) per SSE block.
    """
    method = method.upper()
    if method not in ("GET", "POST"):
        raise ValueError("Only GET and POST supported")

    headers = {"Accept": "text/event-stream"}
    req_kwargs = {"timeout": timeout, "headers": headers}
    if method == "POST" and payload is not None:
        req_kwargs["json"] = payload
    elif method == "GET" and payload is not None:
        req_kwargs["params"] = payload

    with httpx.stream(method, url, **req_kwargs) as r:
        r.raise_for_status()
        data_buf = []
        event_name = None
        for line in r.iter_lines():
            if line is None:
                continue
            line = line.rstrip("\r")
            if line == "":
                # dispatch one SSE event
                if data_buf:
                    yield (event_name, "\n".join(data_buf))
                data_buf.clear()
                event_name = None
                continue
            if line.startswith(":"):
                continue                          # comment / keep-alive
            if line.startswith("event:"):
                event_name = line[6:].lstrip()
                continue
            if line.startswith("data:"):
                v = line[5:]
                if v.startswith(" "): 
                    v = v[1:]
                data_buf.append(v)
                continue
            # ignore other fields (id:, retry:) for now
