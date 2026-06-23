"""LLM utility functions — shared by PM and Chaos agents."""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

_llm: ChatAnthropic | None = None


def get_llm(temperature: float = 0.1) -> ChatAnthropic:
    """获取 LLM 实例（延迟创建，确保 API Key 已加载）"""
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
        )
    return _llm


def extract_text(content: Any) -> str:
    """从 LLM 响应中提取纯文本（兼容 Anthropic 和 DashScope/qwen 格式）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text" and "text" in block:
                    parts.append(block["text"])
                elif block_type == "thinking":
                    continue
                elif "text" in block:
                    parts.append(block["text"])
        return "".join(parts) if parts else ""
    return str(content) if content else ""