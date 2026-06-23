"""PM Agent workers — clarifier, analyzer, and PRD generator."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from .registry import WorkerRegistry
from .pm_state import PMState, ClarificationRound
from .state import WorkerResult

load_dotenv()


def _get_llm(temperature: float = 0.7) -> ChatAnthropic:
    """Get LLM instance."""
    return ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
    )


def _extract_text(content: Any) -> str:
    """Extract text from LLM response (handles both str and list formats)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ═══════════════════════════════════════════════════════════
#  Clarifier Worker — asks targeted questions
# ═══════════════════════════════════════════════════════════

@WorkerRegistry.register(
    name="pm_clarifier",
    description=(
        "你是产品经理助手，擅长通过提问来澄清模糊的需求。"
        "你会分析用户的需求描述，识别关键缺失信息，并提出有针对性的问题。"
        "每轮最多提出3个问题，问题应该具体、有引导性，并提供选项参考。"
    ),
    keywords=["澄清", "提问", "clarify", "question"],
)
async def clarifier_worker_node(state: PMState) -> dict[str, Any]:
    """Clarifier worker — generates targeted clarification questions."""

    context_parts = [f"原始需求: {state.user_query}"]
    for round_data in state.clarification_rounds:
        context_parts.append(f"\n第{round_data.round_number}轮澄清:")
        for i, (q, a) in enumerate(zip(round_data.questions, round_data.answers), 1):
            context_parts.append(f"  Q{i}: {q}")
            context_parts.append(f"  A{i}: {a}")

    context = "\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位资深产品经理，拥有15年互联网产品经验。\n"
         "你的任务是通过精准的提问来澄清用户的需求。\n\n"
         "提问规则:\n"
         "1. 每轮最多提出3个问题\n"
         "2. 问题要具体、有引导性\n"
         "3. 为每个问题提供2-4个选项参考\n"
         "4. 避免重复已经问过的问题\n"
         "5. 聚焦于最关键的缺失信息\n\n"
         "输出格式（严格JSON）:\n"
         "{{\n"
         '  "questions": [\n'
         "    {{\n"
         '      "question": "问题文本",\n'
         '      "options": ["选项1", "选项2", "选项3"]\n'
         "    }}\n"
         "  ]\n"
         "}}\n\n"
         "注意：只输出JSON，不要包含其他文字。"),
        ("human", "{context}\n\n请提出下一轮澄清问题。"),
    ])

    llm = _get_llm(temperature=0.7)
    response = await (prompt | llm).ainvoke({"context": context})
    text = _extract_text(response.content)
    parsed = _parse_json_response(text)

    questions_data = parsed.get("questions", [])
    question_texts = [q.get("question", "") for q in questions_data if q.get("question")]

    user_answers = interrupt({
        "type": "clarification_required",
        "questions": questions_data,
        "round_number": len(state.clarification_rounds) + 1,
    })

    if isinstance(user_answers, list):
        answer_texts = user_answers
    else:
        answer_texts = [str(user_answers)]

    new_round = ClarificationRound(
        round_number=len(state.clarification_rounds) + 1,
        questions=question_texts,
        answers=answer_texts,
    )

    return {
        "clarification_rounds": state.clarification_rounds + [new_round],
        "current_questions": [],
        "current_answers": [],
        "current_step": "orchestrator",
        "messages": state.messages + [
            {"role": "assistant", "content": f"提出澄清问题: {question_texts}"},
            {"role": "user", "content": f"用户回答: {answer_texts}"},
        ],
    }


# ═══════════════════════════════════════════════════════════
#  Analyzer Worker — structured requirements analysis
# ═══════════════════════════════════════════════════════════

@WorkerRegistry.register(
    name="pm_analyzer",
    description=(
        "你是需求分析专家，擅长从模糊需求中提炼结构化信息。\n"
        "你会分析收集到的所有信息，生成结构化的需求分析报告，包括:\n"
        "- 目标用户画像\n"
        "- 功能需求列表（带优先级P0/P1/P2）\n"
        "- 非功能性需求\n"
        "- 风险识别\n"
        "- 推荐技术栈"
    ),
    keywords=["分析", "需求分析", "analyze", "requirements"],
)
async def analyzer_worker_node(state: PMState) -> dict[str, Any]:
    """Analyzer worker — performs structured requirements analysis."""

    context_parts = [f"## 原始需求\n{state.user_query}\n"]
    if state.clarification_rounds:
        context_parts.append("## 澄清问答记录\n")
        for round_data in state.clarification_rounds:
            context_parts.append(f"### 第{round_data.round_number}轮\n")
            for i, (q, a) in enumerate(zip(round_data.questions, round_data.answers), 1):
                context_parts.append(f"**Q{i}:** {q}\n")
                context_parts.append(f"**A{i}:** {a}\n\n")

    context = "\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位资深产品经理，拥有15年互联网产品经验。\n"
         "你的任务是将收集到的需求信息整理成结构化的分析报告。\n\n"
         "输出格式（Markdown）:\n\n"
         "## 1. 目标用户\n"
         "[描述目标用户群体、用户画像]\n\n"
         "## 2. 核心问题\n"
         "[描述产品要解决的核心问题]\n\n"
         "## 3. 功能需求\n\n"
         "### P0 - 必须有（MVP）\n"
         "- **功能名称**: 功能描述\n"
         "  - 用户故事: 作为[角色]，我希望[功能]，以便[价值]\n"
         "  - 验收标准: [具体的验收条件]\n\n"
         "### P1 - 应该有\n"
         "- **功能名称**: 功能描述\n\n"
         "### P2 - 可以有\n"
         "- **功能名称**: 功能描述\n\n"
         "## 4. 非功能性需求\n"
         "- 性能要求\n"
         "- 安全要求\n"
         "- 可用性要求\n\n"
         "## 5. 风险识别\n"
         "- [潜在风险和应对策略]\n\n"
         "## 6. 推荐技术栈\n"
         "- 前端: [框架选择及理由]\n"
         "- 后端: [框架选择及理由]\n"
         "- 数据库: [选择及理由]\n"
         "- 部署: [建议]\n\n"
         "## 7. 参考产品/竞品\n"
         "- [相关参考产品]"),
        ("human", "请基于以下信息生成结构化分析报告:\n\n{context}"),
    ])

    llm = _get_llm(temperature=0.3)
    response = await (prompt | llm).ainvoke({"context": context})
    analysis_result = _extract_text(response.content)

    return {
        "analysis_result": analysis_result,
        "current_step": "prd_generator",
        "worker_results": {
            "pm_analyzer": WorkerResult(
                worker_name="pm_analyzer",
                status="success",
                result=analysis_result,
                tools_used=[],
            )
        },
        "completed_tasks": state.completed_tasks + ["analyze_requirements"],
        "messages": state.messages + [
            {"role": "assistant", "content": "完成需求分析，准备生成PRD文档。"},
        ],
    }


# ═══════════════════════════════════════════════════════════
#  PRD Generator Worker — generates structured PRD document
# ═══════════════════════════════════════════════════════════

@WorkerRegistry.register(
    name="pm_prd_generator",
    description=(
        "你是PRD文档专家，擅长将需求分析结果转化为结构化的产品需求文档。\n"
        "你会生成完整的PRD文档，包括:\n"
        "- 产品概述\n"
        "- 功能需求（带用户故事和验收标准）\n"
        "- 非功能性需求\n"
        "- 技术方案建议\n"
        "- 里程碑计划\n"
        "- 范围外说明"
    ),
    keywords=["PRD", "需求文档", "产品文档", "generate"],
)
async def prd_generator_worker_node(state: PMState) -> dict[str, Any]:
    """PRD Generator worker — creates structured PRD document."""

    analysis = state.analysis_result or "无分析结果"

    feedback_context = ""
    if state.prd_feedback:
        feedback_context = f"\n\n## 用户修改意见\n请根据以下反馈修改PRD:\n{state.prd_feedback}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位资深产品经理，拥有15年互联网产品经验。\n"
         "你的任务是基于需求分析结果，生成一份专业、完整的PRD文档。\n\n"
         "PRD文档要求:\n"
         "1. 功能描述必须具体，开发者可以直接据此编码\n"
         "2. 验收标准必须可测试、可量化\n"
         "3. 避免模糊表述（如'良好的用户体验'）\n"
         "4. 使用Markdown格式，结构清晰\n\n"
         "PRD文档结构:\n\n"
         "# [产品名称] - 产品需求文档 (PRD)\n\n"
         "## 1. 概述\n"
         "### 1.1 产品愿景\n"
         "[一句话描述产品愿景]\n\n"
         "### 1.2 目标用户\n"
         "[目标用户群体描述]\n\n"
         "### 1.3 核心价值主张\n"
         "[产品解决什么问题，为什么用户会选择这个产品]\n\n"
         "## 2. 功能需求\n\n"
         "### 2.1 P0 - 核心功能（MVP）\n\n"
         "#### 功能1: [功能名称]\n"
         "- **描述**: [功能详细描述]\n"
         "- **用户故事**: 作为[角色]，我希望[功能]，以便[价值]\n"
         "- **验收标准**:\n"
         "  - [ ] [具体的、可测试的验收条件1]\n"
         "  - [ ] [具体的、可测试的验收条件2]\n"
         "- **优先级**: P0\n\n"
         "### 2.2 P1 - 重要功能\n"
         "[同上格式]\n\n"
         "### 2.3 P2 - 锦上添花\n"
         "[同上格式]\n\n"
         "## 3. 非功能性需求\n"
         "### 3.1 性能要求\n"
         "- 页面加载时间 < 2秒\n"
         "- API响应时间 < 500ms\n"
         "- 支持[具体数字]并发用户\n\n"
         "### 3.2 安全要求\n"
         "- [具体的安全要求]\n\n"
         "### 3.3 可用性要求\n"
         "- 系统可用性 > 99.9%\n"
         "- [其他可用性要求]\n\n"
         "## 4. 技术方案建议\n"
         "### 4.1 推荐技术栈\n"
         "| 层级 | 技术选择 | 理由 |\n"
         "|------|----------|------|\n"
         "| 前端 | [框架] | [理由] |\n"
         "| 后端 | [框架] | [理由] |\n"
         "| 数据库 | [类型] | [理由] |\n\n"
         "### 4.2 架构建议\n"
         "[简要描述推荐架构]\n\n"
         "## 5. 里程碑计划\n"
         "| 阶段 | 内容 | 预计周期 | 交付物 |\n"
         "|------|------|----------|--------|\n"
         "| MVP | [核心功能] | [周数] | [交付物] |\n"
         "| V1.0 | [完整功能] | [周数] | [交付物] |\n\n"
         "## 6. 范围外（Out of Scope）\n"
         "- [明确不在本期范围内的功能]\n\n"
         "## 7. 开放问题\n"
         "- [待解决的问题]\n\n"
         "## 8. 附录\n"
         "### 8.1 术语表\n"
         "| 术语 | 定义 |\n"
         "|------|------|\n"),
        ("human", "请基于以下需求分析生成完整的PRD文档:\n\n{analysis}{feedback}"),
    ])

    llm = _get_llm(temperature=0.3)
    response = await (prompt | llm).ainvoke({
        "analysis": analysis,
        "feedback": feedback_context,
    })
    prd_document = _extract_text(response.content)

    return {
        "prd_document": prd_document,
        "current_step": "summary",
        "worker_results": {
            **state.worker_results,
            "pm_prd_generator": WorkerResult(
                worker_name="pm_prd_generator",
                status="success",
                result=prd_document,
                tools_used=[],
            )
        },
        "completed_tasks": state.completed_tasks + ["generate_prd_document"],
        "messages": state.messages + [
            {"role": "assistant", "content": "PRD文档生成完成，等待用户确认。"},
        ],
    }