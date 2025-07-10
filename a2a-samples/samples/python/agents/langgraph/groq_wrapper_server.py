"""
Groq → OpenAI-compatible proxy with JSON-schema support
(c) 2025 – free to reuse under MIT licence
"""

import os, json, re, asyncio
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── cfg ──────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_URL: str = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1")
TIMEOUT = 90.0

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = httpx.AsyncClient(timeout=TIMEOUT)
app = FastAPI(title="Groq JSON/Schema wrapper")

# ── helpers ──────────────────────────────────────────────────────────────────
_SCHEMA_PROMPT = (
    "You MUST reply with pure JSON **only** (no markdown). "
    'If you cannot, reply with {"status":"error","message":"Unable to comply"}.'
)

def _inject_json_schema_prompt(payload: Dict[str, Any]) -> None:
    """
    Translate OpenAI-style `response_format` into a system prompt Groq can follow.
    Supports:
      {"type":"json_object"}
      {"type":"json_schema","schema":{…}}
    """
    rf = payload.pop("response_format", None)
    if not rf:
        return

    if rf.get("type") == "schemajson_":
        # OpenAI spec uses `"schema"`, not `"json_schema"` :contentReference[oaicite:0]{index=0}
        schema = json.dumps(rf["json_schema"], separators=(",", ":"))
        instruction = (
            f"{_SCHEMA_PROMPT} The response MUST validate against this schema:\n{schema}"
        )
    else:  # json_object or anything else
        instruction = _SCHEMA_PROMPT

    payload["messages"] = [{"role": "system", "content": instruction}] + payload["messages"]

_JSON_RE = re.compile(r"\{.*?\}", re.S)  # make it non-greedy :contentReference[oaicite:1]{index=1}

def _extract_first_json(blob: str) -> Optional[dict]:
    """Return the first {...} block in `blob`, or None."""
    match = _JSON_RE.search(blob)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def _unpack_content(choice: Dict[str, Any]) -> Union[str, dict, None]:
    """
    Safely grab whatever the assistant returned:
    - normal text:      returns str
    - tool/function:    returns dict under "tool_calls"/"function_call"
    - nothing:          returns None
    Avoids KeyError crash :contentReference[oaicite:2]{index=2}
    """
    msg = choice.get("message", {})
    if "content" in msg:
        return msg["content"]
    if "tool_calls" in msg:
        return msg["tool_calls"]
    if "function_call" in msg:
        return msg["function_call"]
    return None

# ── streaming proxy ──────────────────────────────────────────────────────────
async def _relay_stream(resp: httpx.Response) -> AsyncIterator[bytes]:
    """
    Yield the upstream SSE stream unchanged.
    """
    async for chunk in resp.aiter_bytes():
        yield chunk

# ── main endpoint ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body: Dict[str, Any] = await request.json()
    stream = bool(body.get("stream", False))   # preserve caller preference
    _inject_json_schema_prompt(body)

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # pass the same stream flag to Groq
    url = f"{GROQ_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    if stream:                      # ── streaming path
        async def relay():
            async with client.stream("POST", url, headers=headers, json=body) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
        return StreamingResponse(relay(), media_type="text/event-stream")

    # ── non-streaming path
    resp = await client.post(url, headers=headers, json=body)

    if resp.status_code != 200:
        detail = await resp.aread()
        raise HTTPException(status_code=resp.status_code, detail=detail.decode())

    # ── streaming mode: just relay ───────────────────────────────────────────
    if stream:
        return StreamingResponse(
            _relay_stream(resp),
            media_type="text/event-stream",
            status_code=200,
        )

    # ── sync mode: post-process one-shot response ────────────────────────────
    data = resp.json()
    choice0 = data["choices"][0]

    raw_content = _unpack_content(choice0)
    parsed = _extract_first_json(raw_content) if isinstance(raw_content, str) else None
    refusal = None
    if body.get("response_format") and parsed is None:
        refusal = {"status": "error", "message": "LLM did not return standalone JSON"}

    # expose helpers under OpenAI-style additional_kwargs
    choice0.setdefault("message", {}).setdefault("additional_kwargs", {})
    choice0["message"]["additional_kwargs"].update({"parsed": parsed, "refusal": refusal})

    return JSONResponse(content=data, status_code=200)
