from typing import Optional
import requests
import json

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
        lines.append(f"id:{id}")
    if event is not None:
        lines.append(f"event:{event}")
    if retry is not None:
        lines.append(f"retry:{retry}")
    for line in data.splitlines() or [""]:
        lines.append(f"data:{line}")
    lines.append("")  # blank line to terminate the message
    return ("\n".join(lines) + "\n").encode("utf-8")
