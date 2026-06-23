"""Chaos testing target agent connector — internal + external HTTP calls."""

import time
import httpx
from typing import Any

INTERNAL_BASE = "http://localhost:8000"


async def call_internal_agent(target: str, user_input: str, thread_id: str | None = None) -> dict:
    """Call the PM agent via HTTP API and return response + metadata."""
    import uuid
    start = time.time()
    tid = thread_id or str(uuid.uuid4())

    if target == "pm":
        url = f"{INTERNAL_BASE}/api/pm/stream"
    else:
        raise ValueError(f"Unknown internal target: {target}")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json={"query": user_input, "thread_id": tid})
        events = resp.text.split("\n")
        response_text = _extract_pm_sse_response(events)
        return {
            "response": response_text or "[PM did not produce output]",
            "response_time_ms": int((time.time() - start) * 1000),
            "raw": events,
        }


async def call_external_agent(config: dict, user_input: str) -> dict:
    """Call an external agent via HTTP proxy (OpenAI-compatible API)."""
    start = time.time()
    url = config.get("url", "")
    api_key = config.get("api_key", "")
    model = config.get("model", "gpt-3.5-turbo")
    timeout = config.get("timeout", 60)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_input}],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "response": content,
            "response_time_ms": int((time.time() - start) * 1000),
            "raw": data,
        }


def _extract_pm_sse_response(events: list[str]) -> str:
    """Extract final response text from PM SSE events."""
    for line in reversed(events):
        if "prd_document" in line or "final_result" in line:
            return line
    return ""