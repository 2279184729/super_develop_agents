"""Unit tests for PM Workers — clarifier, analyzer, prd_generator."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import pm_workers as pm_wkr_mod
from src.agent.pm_state import PMState, ClarificationRound


def _make_mock_chain(response_content: str):
    """Create mock LLM chain."""
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    mock_chain.__or__ = MagicMock(return_value=mock_chain)
    return mock_chain


class TestClarifierWorker:
    """澄清 Worker 单元测试"""

    @pytest.mark.asyncio
    async def test_generates_questions_with_options(self, clean_registry):
        """澄清 Worker 应生成带选项的问题"""
        questions_json = json.dumps({
            "questions": [
                {"question": "目标用户是谁?", "options": ["学生", "白领", "老年人"]},
                {"question": "核心功能?", "options": ["点餐", "配送", "评价"]},
            ]
        })
        mock_chain = _make_mock_chain(questions_json)

        # Mock interrupt to return user answers
        mock_answers = ["白领", "点餐"]

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt, \
             patch.object(pm_wkr_mod, "interrupt", return_value=mock_answers) as mock_interrupt:
            mock_prompt.from_messages.return_value = mock_chain

            # Register the worker
            from src.agent.pm_workers import clarifier_worker_node

            state = PMState(
                user_query="我想做一个外卖App",
                messages=[{"role": "user", "content": "我想做一个外卖App"}],
            )
            result = await clarifier_worker_node(state)

            # Verify interrupt was called
            mock_interrupt.assert_called_once()
            interrupt_data = mock_interrupt.call_args[0][0]
            assert interrupt_data["type"] == "clarification_required"
            assert len(interrupt_data["questions"]) == 2

            # Verify result
            assert len(result["clarification_rounds"]) == 1
            assert result["current_step"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_max_three_questions(self, clean_registry):
        """每轮最多3个问题"""
        questions_json = json.dumps({
            "questions": [
                {"question": "Q1", "options": ["A", "B"]},
                {"question": "Q2", "options": ["C", "D"]},
                {"question": "Q3", "options": ["E", "F"]},
            ]
        })
        mock_chain = _make_mock_chain(questions_json)

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt, \
             patch.object(pm_wkr_mod, "interrupt", return_value=["A1", "A2", "A3"]):
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import clarifier_worker_node

            state = PMState(user_query="测试", messages=[])
            result = await clarifier_worker_node(state)

            interrupt_data = None
            # Get the interrupt call data
            for call in pm_wkr_mod.interrupt.call_args_list:
                interrupt_data = call[0][0]

            assert len(interrupt_data["questions"]) == 3

    @pytest.mark.asyncio
    async def test_handles_string_answer(self, clean_registry):
        """单个字符串答案应被转为列表"""
        questions_json = json.dumps({
            "questions": [{"question": "Q1", "options": []}]
        })
        mock_chain = _make_mock_chain(questions_json)

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt, \
             patch.object(pm_wkr_mod, "interrupt", return_value="单个答案"):
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import clarifier_worker_node

            state = PMState(user_query="测试", messages=[])
            result = await clarifier_worker_node(state)

            new_round = result["clarification_rounds"][0]
            assert new_round.answers == ["单个答案"]


class TestAnalyzerWorker:
    """分析 Worker 单元测试"""

    @pytest.mark.asyncio
    async def test_generates_structured_analysis(self, clean_registry):
        """分析 Worker 应生成结构化分析报告"""
        analysis_text = """## 1. 目标用户
年轻白领

## 2. 核心问题
外卖点餐效率低

## 3. 功能需求
### P0 - 必须有（MVP）
- **快速点餐**: 3步完成下单

## 4. 非功能性需求
- 页面加载 < 2秒

## 5. 风险识别
- 配送延迟

## 6. 推荐技术栈
- 前端: React
"""
        mock_chain = _make_mock_chain(analysis_text)

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import analyzer_worker_node

            state = PMState(
                user_query="外卖App",
                messages=[],
                clarification_rounds=[
                    ClarificationRound(
                        round_number=1,
                        questions=["目标用户?"],
                        answers=["年轻白领"],
                    ),
                ],
            )
            result = await analyzer_worker_node(state)

            assert "目标用户" in result["analysis_result"]
            assert result["current_step"] == "prd_generator"
            assert len(result["completed_tasks"]) == 1

    @pytest.mark.asyncio
    async def test_includes_clarification_history(self, clean_registry):
        """分析应包含所有澄清历史"""
        mock_chain = _make_mock_chain("## 分析报告")

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import analyzer_worker_node

            state = PMState(
                user_query="测试",
                messages=[],
                clarification_rounds=[
                    ClarificationRound(round_number=1, questions=["Q1"], answers=["A1"]),
                    ClarificationRound(round_number=2, questions=["Q2"], answers=["A2"]),
                ],
            )
            result = await analyzer_worker_node(state)

            assert result["analysis_result"] == "## 分析报告"


class TestPRDGeneratorWorker:
    """PRD 生成 Worker 单元测试"""

    @pytest.mark.asyncio
    async def test_generates_prd_document(self, clean_registry):
        """PRD Worker 应生成完整的 PRD 文档"""
        prd_text = """# 外卖App - 产品需求文档 (PRD)

## 1. 概述
### 1.1 产品愿景
让外卖点餐更快捷

## 2. 功能需求
### 2.1 P0 - 核心功能（MVP）
#### 功能1: 快速点餐
- **描述**: 3步完成下单
- **验收标准**:
  - [ ] 下单时间 < 30秒

## 5. 里程碑计划
| 阶段 | 内容 | 预计周期 |
|------|------|----------|
| MVP | 核心功能 | 4周 |
"""
        mock_chain = _make_mock_chain(prd_text)

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import prd_generator_worker_node

            state = PMState(
                user_query="外卖App",
                messages=[],
                analysis_result="## 需求分析\n目标用户: 年轻白领",
            )
            result = await prd_generator_worker_node(state)

            assert "产品需求文档" in result["prd_document"]
            assert result["current_step"] == "summary"
            assert len(result["completed_tasks"]) == 1

    @pytest.mark.asyncio
    async def test_handles_feedback(self, clean_registry):
        """有修改反馈时应在 prompt 中包含"""
        mock_chain = _make_mock_chain("# 修改后的 PRD")

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import prd_generator_worker_node

            state = PMState(
                user_query="测试",
                messages=[],
                analysis_result="分析报告",
                prd_feedback="增加社交功能",
            )
            result = await prd_generator_worker_node(state)

            assert result["prd_document"] == "# 修改后的 PRD"

    @pytest.mark.asyncio
    async def test_prd_contains_required_sections(self, clean_registry):
        """PRD 应包含必需的章节"""
        prd_text = """# 产品PRD

## 1. 概述
### 1.1 产品愿景
愿景描述

## 2. 功能需求
### 2.1 P0 - 核心功能（MVP）
#### 功能1: 核心功能
- **用户故事**: 作为用户，我希望...
- **验收标准**:
  - [ ] 标准1

## 3. 非功能性需求
- 性能要求

## 4. 技术方案建议
| 层级 | 技术选择 | 理由 |
|------|----------|------|
| 前端 | React | 生态完善 |

## 5. 里程碑计划
| 阶段 | 内容 | 预计周期 | 交付物 |
|------|------|----------|--------|
| MVP | 核心 | 4周 | v0.1 |

## 6. 范围外（Out of Scope）
- 国际化
"""
        mock_chain = _make_mock_chain(prd_text)

        with patch.object(pm_wkr_mod, "_get_llm", return_value=mock_chain), \
             patch.object(pm_wkr_mod, "ChatPromptTemplate") as mock_prompt:
            mock_prompt.from_messages.return_value = mock_chain

            from src.agent.pm_workers import prd_generator_worker_node

            state = PMState(
                user_query="测试",
                messages=[],
                analysis_result="分析",
            )
            result = await prd_generator_worker_node(state)

            prd = result["prd_document"]
            assert "概述" in prd
            assert "功能需求" in prd
            assert "验收标准" in prd
            assert "非功能性需求" in prd
            assert "技术方案" in prd
            assert "里程碑" in prd
            assert "范围外" in prd
