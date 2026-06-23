"""Shared test fixtures for PM + Chaos agents tests."""

import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
#  Pre-import mocking
# ═══════════════════════════════════════════════════════════

def _ensure_module(name: str) -> types.ModuleType:
    """确保 sys.modules 中存在指定模块，不存在则创建 mock"""
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


_anth_mod = _ensure_module("langchain_anthropic")
if not hasattr(_anth_mod, "ChatAnthropic") or not isinstance(_anth_mod.ChatAnthropic, MagicMock):
    _anth_mod.ChatAnthropic = MagicMock()

_agents_mod = _ensure_module("langchain.agents")
for attr in ("AgentExecutor", "create_tool_calling_agent"):
    if not hasattr(_agents_mod, attr) or not isinstance(getattr(_agents_mod, attr), MagicMock):
        setattr(_agents_mod, attr, MagicMock())


# ═══════════════════════════════════════════════════════════
#  Mock LLM 响应
# ═══════════════════════════════════════════════════════════

MOCK_ORCHESTRATOR_RESPONSE = '''{
  "tasks": [
    {"task": "研究: 测试查询", "worker": "research_worker", "priority": 1}
  ]
}'''

MOCK_SUMMARY_RESPONSE = "这是汇总结果。"


class MockAIMessage:
    """模拟 LangChain AIMessage"""
    def __init__(self, content: str):
        self.content = content


class MockLLMChain:
    """模拟 LLM Chain（支持 ainvoke 和 invoke）"""
    def __init__(self, response_content: str):
        self._response = response_content

    async def ainvoke(self, inputs: dict[str, Any]) -> MockAIMessage:  # noqa: ARG002
        return MockAIMessage(self._response)

    def invoke(self, inputs: dict[str, Any]) -> MockAIMessage:  # noqa: ARG002
        return MockAIMessage(self._response)

    async def __or__(self, other):
        return self

    def __or__(self, other):  # noqa: F811
        return self


@pytest.fixture
def mock_llm_response():
    """返回指定内容的 mock LLM 响应"""
    def _make(content: str):
        return MockLLMChain(content)
    return _make


# ═══════════════════════════════════════════════════════════
#  FastAPI TestClient
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def test_client():
    """FastAPI 测试客户端"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    test_app = FastAPI()

    from src.api.main import app as main_app
    for route in main_app.routes:
        test_app.routes.append(route)

    client = TestClient(test_app)
    yield client


# ═══════════════════════════════════════════════════════════
#  Worker Registry 清理
# ═══════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理 Worker 注册表"""
    from src.agent.registry import WorkerRegistry
    WorkerRegistry.clear()
    yield
    WorkerRegistry.clear()