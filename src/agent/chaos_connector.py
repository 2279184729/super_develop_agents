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
    """Extract meaningful response text from PM Agent SSE events.

    Parses the SSE protocol (event:/data: pairs), accumulates streaming tokens,
    and falls back to structured events (clarification, analysis, PRD, etc.).
    """
    import json as _json

    token_parts: list[str] = []
    structured: dict[str, str] = {}

    i = 0
    while i < len(events):
        line = events[i].strip()
        if line.startswith("event: "):
            event_type = line[7:]
            data = ""
            if i + 1 < len(events) and events[i + 1].strip().startswith("data: "):
                data = events[i + 1].strip()[6:]
                i += 1
            if event_type == "token" and data:
                try:
                    payload = _json.loads(data)
                    content = payload.get("content", "")
                    if isinstance(content, str):
                        token_parts.append(content)
                except (_json.JSONDecodeError, TypeError):
                    pass
            elif data:
                try:
                    payload = _json.loads(data)
                except (_json.JSONDecodeError, TypeError):
                    payload = {}
                if event_type == "prd_ready":
                    prd = payload.get("prd_document", "")
                    if prd:
                        structured["prd"] = prd
                elif event_type == "done":
                    final = payload.get("final_result", "")
                    prd = payload.get("prd_document", "")
                    if prd:
                        structured["prd"] = prd
                    elif final:
                        structured["final"] = final
                elif event_type == "analysis_done":
                    analysis = payload.get("analysis_result", "")
                    if analysis:
                        structured["analysis"] = analysis
                elif event_type == "clarification_required":
                    questions = payload.get("questions", [])
                    if questions:
                        q_texts = []
                        for q in questions:
                            if isinstance(q, dict):
                                q_texts.append(q.get("question", str(q)))
                            else:
                                q_texts.append(str(q))
                        structured["clarification"] = " | ".join(q_texts)
                elif event_type == "orchestrator_decision":
                    decision = payload.get("decision", "")
                    sufficiency = payload.get("information_sufficiency", 0)
                    structured["decision"] = (
                        f"decision={decision}, sufficiency={sufficiency:.0%}"
                    )
        i += 1

    if token_parts:
        return "".join(token_parts)
    for key in ("prd", "final", "analysis", "clarification", "decision"):
        if key in structured:
            return structured[key]
    return ""