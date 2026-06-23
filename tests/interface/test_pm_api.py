"""Interface tests for PM Agent API endpoints."""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.agent.pm_state import PMState


class TestPMRequestModels:
    """PM 请求模型测试"""

    def test_pm_request_defaults(self):
        from src.api.main import PMRequest
        req = PMRequest(query="我想做一个App")
        assert req.query == "我想做一个App"
        assert req.thread_id is None

    def test_pm_request_with_thread(self):
        from src.api.main import PMRequest
        req = PMRequest(query="测试", thread_id="thread-123")
        assert req.thread_id == "thread-123"

    def test_pm_answer_request(self):
        from src.api.main import PMAnswerRequest
        req = PMAnswerRequest(answers=["答案1", "答案2", "答案3"])
        assert len(req.answers) == 3
        assert req.answers[0] == "答案1"

    def test_pm_confirm_request_approve(self):
        from src.api.main import PMConfirmRequest
        req = PMConfirmRequest(approve=True)
        assert req.approve is True
        assert req.feedback is None

    def test_pm_confirm_request_with_feedback(self):
        from src.api.main import PMConfirmRequest
        req = PMConfirmRequest(approve=True, feedback="增加功能X")
        assert req.feedback == "增加功能X"


class TestPMAPIEndpoints:
    """PM API 端点测试"""

    def test_health_check(self, test_client):
        """健康检查端点应正常"""
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_pm_state_not_found(self, test_client):
        """不存在的 thread 应返回 404"""
        with patch("src.api.main.get_pm_graph_state", new_callable=AsyncMock, return_value=None):
            resp = test_client.get("/api/pm/state/nonexistent-thread")
            assert resp.status_code == 404

    def test_pm_download_not_found(self, test_client):
        """下载不存在的 PRD 应返回 404"""
        with patch("src.api.main.get_pm_graph_state", new_callable=AsyncMock, return_value=None):
            resp = test_client.get("/api/pm/download/nonexistent-thread")
            assert resp.status_code == 404

    def test_pm_download_no_prd(self, test_client):
        """状态存在但无 PRD 时应返回 404"""
        mock_state = {"analysis_result": "分析报告", "prd_document": None}
        with patch("src.api.main.get_pm_graph_state", new_callable=AsyncMock, return_value=mock_state):
            resp = test_client.get("/api/pm/download/some-thread")
            assert resp.status_code == 404

    def test_pm_download_prd(self, test_client):
        """有 PRD 时应返回 Markdown 文件"""
        mock_state = {
            "analysis_result": "分析报告",
            "prd_document": "# PRD\n## 功能需求\n...",
        }
        with patch("src.api.main.get_pm_graph_state", new_callable=AsyncMock, return_value=mock_state):
            resp = test_client.get("/api/pm/download/test-thread-12345")
            assert resp.status_code == 200
            assert "text/markdown" in resp.headers["content-type"]
            assert "PRD_test-thr.md" in resp.headers["content-disposition"]
            assert "PRD" in resp.text

    def test_pm_stream_returns_sse(self, test_client):
        """PM stream 端点应返回 SSE 流"""
        async def mock_stream(query, thread_id):
            yield "event: connected\ndata: {\"thread_id\": \"test-123\"}\n\n"
            yield "event: status\ndata: {\"node\": \"orchestrator\", \"status\": \"started\"}\n\n"

        with patch("src.api.main.run_pm_stream", side_effect=mock_stream):
            resp = test_client.post(
                "/api/pm/stream",
                json={"query": "我想做一个App"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_pm_answer_returns_sse(self, test_client):
        """提交答案应返回 SSE 流"""
        async def mock_answer(thread_id, answers):
            yield "event: status\ndata: {\"node\": \"resumed\"}\n\n"

        with patch("src.api.main.answer_pm_questions", side_effect=mock_answer):
            resp = test_client.post(
                "/api/pm/answer/test-thread",
                json={"answers": ["白领", "点餐"]},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_pm_confirm_returns_sse(self, test_client):
        """确认 PRD 应返回 SSE 流"""
        async def mock_confirm(thread_id, approve, feedback):
            yield "event: status\ndata: {\"node\": \"confirmed\"}\n\n"

        with patch("src.api.main.confirm_prd", side_effect=mock_confirm):
            resp = test_client.post(
                "/api/pm/confirm/test-thread",
                json={"approve": True},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_pm_state_returns_state(self, test_client):
        """获取 PM 状态应返回状态数据"""
        mock_state = {
            "user_query": "测试",
            "pm_decision": "need_clarification",
            "information_sufficiency": 0.3,
            "__interrupted": True,
        }
        with patch("src.api.main.get_pm_graph_state", new_callable=AsyncMock, return_value=mock_state):
            resp = test_client.get("/api/pm/state/test-thread")
            assert resp.status_code == 200
            data = resp.json()
            assert data["pm_decision"] == "need_clarification"
            assert data["__interrupted"] is True
