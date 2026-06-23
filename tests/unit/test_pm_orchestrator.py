"""Unit tests for PM Orchestrator and Summary nodes."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import pm_orchestrator as pm_orch_mod
from src.agent.pm_state import PMState, ClarificationRound


def _make_mock_chain(response_content: str):
    """Create mock LLM chain with specified response."""
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_chain.__or__ = MagicMock(return_value=mock_chain)
    return mock_chain


class TestPMOrchestratorNode:
    """PM 编排者节点单元测试"""

    @pytest.mark.asyncio
    async def test_first_call_decides_need_clarification(self, clean_registry):
        """首次调用时，信息不足应判定 need_clarification"""
        decision_json = json.dumps({
            "decision": "need_clarification",
            "information_sufficiency": 0.2,
            "reasoning": "目标用户和核心功能都不明确",
        })
        mock_chain = _make_mock_chain(decision_json)

        with patch.object(pm_orch_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_orch_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            state = PMState(
                user_query="我想做一个App",
                messages=[{"role": "user", "content": "我想做一个App"}],
            )
            result = await pm_orch_mod.pm_orchestrator_node(state)

            assert result["pm_decision"] == "need_clarification"
            assert result["current_step"] == "clarifier"
            assert result["information_sufficiency"] == 0.2

    @pytest.mark.asyncio
    async def test_sufficient_info_decides_ready_to_generate(self, clean_registry):
        """信息充足时应判定 ready_to_generate"""
        decision_json = json.dumps({
            "decision": "ready_to_generate",
            "information_sufficiency": 0.9,
            "reasoning": "目标用户、核心功能、技术约束都已明确",
        })
        mock_chain = _make_mock_chain(decision_json)

        with patch.object(pm_orch_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_orch_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            state = PMState(
                user_query="我想做一个外卖App",
                messages=[{"role": "user", "content": "我想做一个外卖App"}],
                clarification_rounds=[
                    ClarificationRound(
                        round_number=1,
                        questions=["目标用户?", "核心功能?", "技术约束?"],
                        answers=["年轻白领", "点餐配送", "React + Node.js"],
                    ),
                ],
            )
            result = await pm_orch_mod.pm_orchestrator_node(state)

            assert result["pm_decision"] == "ready_to_generate"
            assert result["current_step"] == "analyzer"
            assert result["information_sufficiency"] == 0.9

    @pytest.mark.asyncio
    async def test_max_rounds_forces_ready(self, clean_registry):
        """达到最大澄清轮次时应强制 ready_to_generate"""
        state = PMState(
            user_query="测试需求",
            messages=[{"role": "user", "content": "测试需求"}],
            clarification_rounds=[
                ClarificationRound(round_number=1, questions=["Q1"], answers=["A1"]),
                ClarificationRound(round_number=2, questions=["Q2"], answers=["A2"]),
                ClarificationRound(round_number=3, questions=["Q3"], answers=["A3"]),
            ],
            max_clarification_rounds=3,
        )
        # Should not even call LLM
        result = await pm_orch_mod.pm_orchestrator_node(state)

        assert result["pm_decision"] == "ready_to_generate"
        assert result["current_step"] == "analyzer"
        assert result["information_sufficiency"] == 0.8

    @pytest.mark.asyncio
    async def test_records_clarification_context(self, clean_registry):
        """编排者应在 context 中包含所有澄清历史"""
        decision_json = json.dumps({
            "decision": "need_clarification",
            "information_sufficiency": 0.4,
            "reasoning": "还需要更多信息",
        })
        mock_chain = _make_mock_chain(decision_json)

        with patch.object(pm_orch_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_orch_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            state = PMState(
                user_query="测试",
                messages=[],
                clarification_rounds=[
                    ClarificationRound(
                        round_number=1,
                        questions=["目标用户?"],
                        answers=["学生"],
                    ),
                ],
            )
            result = await pm_orch_mod.pm_orchestrator_node(state)

            # Verify the result contains messages
            assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_bad_json_fallback(self, clean_registry):
        """LLM 返回非 JSON 时应使用默认值"""
        mock_chain = _make_mock_chain("这不是JSON")

        with patch.object(pm_orch_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_orch_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            state = PMState(
                user_query="测试",
                messages=[],
            )
            result = await pm_orch_mod.pm_orchestrator_node(state)

            # Default fallback values
            assert result["pm_decision"] == "need_clarification"
            assert result["information_sufficiency"] == 0.5
            assert result["current_step"] == "clarifier"


class TestPMSummaryNode:
    """PM 汇总节点单元测试"""

    @pytest.mark.asyncio
    async def test_summary_combines_analysis_and_prd(self, clean_registry):
        """汇总节点应组合分析结果和PRD文档"""
        state = PMState(
            user_query="测试",
            messages=[],
            analysis_result="## 需求分析\n目标用户: 年轻白领",
            prd_document="# PRD\n## 功能需求\n...",
            clarification_rounds=[
                ClarificationRound(
                    round_number=1,
                    questions=["目标用户?"],
                    answers=["年轻白领"],
                ),
            ],
        )
        result = await pm_orch_mod.pm_summary_node(state)

        assert "需求分析" in result["final_result"]
        assert "PRD 文档" in result["final_result"]
        assert "年轻白领" in result["final_result"]
        assert result["current_step"] == "end"
        assert result["prd_confirmed"] is True

    @pytest.mark.asyncio
    async def test_summary_without_analysis(self, clean_registry):
        """无分析结果时应生成默认消息"""
        state = PMState(
            user_query="测试",
            messages=[],
        )
        result = await pm_orch_mod.pm_summary_node(state)

        assert "未能生成完整的PRD文档" in result["final_result"]
        assert result["current_step"] == "end"

    @pytest.mark.asyncio
    async def test_summary_includes_clarification_record(self, clean_registry):
        """汇总应包含澄清记录"""
        state = PMState(
            user_query="测试",
            messages=[],
            analysis_result="分析报告",
            prd_document="PRD文档",
            clarification_rounds=[
                ClarificationRound(
                    round_number=1,
                    questions=["Q1", "Q2"],
                    answers=["A1", "A2"],
                ),
                ClarificationRound(
                    round_number=2,
                    questions=["Q3"],
                    answers=["A3"],
                ),
            ],
        )
        result = await pm_orch_mod.pm_summary_node(state)

        assert "澄清记录" in result["final_result"]
        assert "第1轮" in result["final_result"]
        assert "第2轮" in result["final_result"]
        assert "Q1" in result["final_result"]
        assert "A3" in result["final_result"]
