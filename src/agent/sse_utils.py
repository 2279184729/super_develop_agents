"""SSE streaming utilities for LangGraph event processing."""

import json
from typing import Any


def sse(event_type: str, data: Any) -> str:
    """构造 SSE 格式字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def extract_output(event: dict[str, Any]) -> dict[str, Any] | None:
    """从 astream_events 的 on_chain_end 事件中提取 output"""
    data = event.get("data", {})
    output = data.get("output")
    if isinstance(output, dict):
        return output
    return None


def extract_stream_token(event: dict[str, Any]) -> str:
    """从流式事件中提取文本 token（兼容 Anthropic 和 DashScope/qwen 格式）"""
    chunk = event.get("data", {}).get("chunk")
    if not chunk or not hasattr(chunk, "content") or not chunk.content:
        return ""

    content = chunk.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "thinking":
                    continue
                if block_type == "signature":
                    continue
                if block_type == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "".join(parts)

    return ""