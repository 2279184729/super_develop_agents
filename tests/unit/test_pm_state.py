"""Unit tests for PMState and ClarificationRound models."""

import pytest

from src.agent.pm_state import PMState, ClarificationRound


class TestClarificationRound:
    """ClarificationRound 模型测试"""

    def test_create_with_defaults(self):
        cr = ClarificationRound(round_number=1)
        assert cr.round_number == 1
        assert cr.questions == []
        assert cr.answers == []

    def test_create_with_data(self):
        cr = ClarificationRound(
            round_number=2,
            questions=["目标用户是谁?", "核心功能是什么?"],
            answers=["年轻白领", "外卖点餐"],
        )
        assert cr.round_number == 2
        assert len(cr.questions) == 2
        assert len(cr.answers) == 2
        assert cr.questions[0] == "目标用户是谁?"
        assert cr.answers[1] == "外卖点餐"

    def test_serialization(self):
        cr = ClarificationRound(
            round_number=1,
            questions=["Q1"],
            answers=["A1"],
        )
        data = cr.model_dump()
        assert data["round_number"] == 1
        assert data["questions"] == ["Q1"]
        assert data["answers"] == ["A1"]

    def test_missing_required_field(self):
        with pytest.raises(Exception):  # noqa: B017
            ClarificationRound()  # round_number is required


class TestPMState:
    """PMState 模型测试"""

    def test_create_with_defaults(self):
        state = PMState(user_query="我想做一个App")
        # Inherited fields from AgentState
        assert state.user_query == "我想做一个App"
        assert state.messages == []
        assert state.worker_results == {}
        assert state.completed_tasks == []
        assert state.current_step == "orchestrator"
        # PM-specific fields
        assert state.clarification_rounds == []
        assert state.current_questions == []
        assert state.current_answers == []
        assert state.max_clarification_rounds == 3
        assert state.pm_decision == "need_clarification"
        assert state.information_sufficiency == 0.0
        assert state.analysis_result is None
        assert state.prd_document is None
        assert state.prd_confirmed is False
        assert state.prd_feedback is None

    def test_create_with_all_pm_fields(self):
        cr = ClarificationRound(round_number=1, questions=["Q1"], answers=["A1"])
        state = PMState(
            user_query="测试需求",
            clarification_rounds=[cr],
            current_questions=["Q2"],
            current_answers=["A2"],
            max_clarification_rounds=5,
            pm_decision="ready_to_generate",
            information_sufficiency=0.85,
            analysis_result="## 分析报告",
            prd_document="# PRD文档",
            prd_confirmed=True,
            prd_feedback="增加功能X",
        )
        assert len(state.clarification_rounds) == 1
        assert state.clarification_rounds[0].round_number == 1
        assert state.pm_decision == "ready_to_generate"
        assert state.information_sufficiency == 0.85
        assert state.analysis_result == "## 分析报告"
        assert state.prd_document == "# PRD文档"
        assert state.prd_confirmed is True
        assert state.prd_feedback == "增加功能X"

    def test_inherits_agent_state_fields(self):
        """PMState 应继承 AgentState 的所有字段"""
        state = PMState(
            user_query="测试",
            messages=[{"role": "user", "content": "测试"}],
            task_plan=["任务1"],
            completed_tasks=["任务1"],
        )
        assert state.user_query == "测试"
        assert len(state.messages) == 1
        assert state.task_plan == ["任务1"]
        assert state.completed_tasks == ["任务1"]

    def test_model_dump(self):
        state = PMState(user_query="测试")
        data = state.model_dump()
        assert isinstance(data, dict)
        assert data["user_query"] == "测试"
        assert "clarification_rounds" in data
        assert "pm_decision" in data
        assert "prd_document" in data

    def test_add_clarification_round(self):
        state = PMState(user_query="测试")
        cr = ClarificationRound(
            round_number=1,
            questions=["Q1", "Q2"],
            answers=["A1", "A2"],
        )
        state.clarification_rounds.append(cr)
        assert len(state.clarification_rounds) == 1
        assert state.clarification_rounds[0].round_number == 1

    def test_multiple_clarification_rounds(self):
        state = PMState(user_query="测试")
        for i in range(1, 4):
            state.clarification_rounds.append(
                ClarificationRound(
                    round_number=i,
                    questions=[f"Q{i}"],
                    answers=[f"A{i}"],
                )
            )
        assert len(state.clarification_rounds) == 3
        assert state.clarification_rounds[-1].round_number == 3
