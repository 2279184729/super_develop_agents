"""PM Agent orchestrator and summary nodes."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from .pm_state import PMState
from .llm_utils import extract_text

load_dotenv()


def _get_llm(temperature: float = 0.1) -> ChatAnthropic:
    """Get LLM instance."""
    return ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
    )


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
#  PM Orchestrator Node — evaluates sufficiency, routes decisions
# ═══════════════════════════════════════════════════════════

async def pm_orchestrator_node(state: PMState) -> dict[str, Any]:
    """PM orchestrator: evaluate information sufficiency → decide clarification or generation."""

    round_count = len(state.clarification_rounds)
    if round_count >= state.max_clarification_rounds:
        return {
            "pm_decision": "ready_to_generate",
            "information_sufficiency": 0.8,
            "current_step": "analyzer",
            "messages": state.messages + [
                {"role": "assistant", "content": f"已完成{round_count}轮澄清，信息充足，准备生成PRD。"},
            ],
        }

    context_parts = [f"原始需求: {state.user_query}"]
    for round_data in state.clarification_rounds:
        context_parts.append(f"\n第{round_data.round_number}轮澄清:")
        for i, (q, a) in enumerate(zip(round_data.questions, round_data.answers), 1):
            context_parts.append(f"  Q{i}: {q}")
            context_parts.append(f"  A{i}: {a}")

    context = "\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个产品经理Agent的编排者。你的任务是评估用户需求的完整程度，决定下一步行动。\n\n"
         "信息充足度评估标准:\n"
         "1. 目标用户群体是否明确\n"
         "2. 核心功能是否清晰\n"
         "3. 业务目标/成功指标是否定义\n"
         "4. 技术约束是否了解\n"
         "5. 优先级是否明确\n\n"
         "输出格式（严格JSON）:\n"
         "{{\n"
         '  "decision": "need_clarification" 或 "ready_to_generate",\n'
         '  "information_sufficiency": 0.0到1.0的浮点数,\n'
         '  "reasoning": "判断理由"\n'
         "}}\n\n"
         "注意：只输出JSON，不要包含其他文字。"),
        ("human",
         "已完成{round_count}/{max_rounds}轮澄清。\n\n"
         "{context}\n\n"
         "请评估信息充足度并决定下一步。"),
    ])

    llm = _get_llm(temperature=0.1)
    response = await (prompt | llm).ainvoke({
        "round_count": round_count,
        "max_rounds": state.max_clarification_rounds,
        "context": context,
    })

    text = extract_text(response.content)
    parsed = _parse_json_response(text)

    decision = parsed.get("decision", "need_clarification")
    sufficiency = float(parsed.get("information_sufficiency", 0.5))
    reasoning = parsed.get("reasoning", "")

    if decision == "ready_to_generate":
        current_step = "analyzer"
        message = f"信息充足度: {sufficiency:.0%}。{reasoning}。准备生成PRD。"
    else:
        current_step = "clarifier"
        message = f"信息充足度: {sufficiency:.0%}。{reasoning}。需要进一步澄清。"

    return {
        "pm_decision": decision,
        "information_sufficiency": sufficiency,
        "current_step": current_step,
        "messages": state.messages + [
            {"role": "assistant", "content": message},
        ],
    }


# ═══════════════════════════════════════════════════════════
#  PM Summary Node — combines analysis + PRD into final output
# ═══════════════════════════════════════════════════════════

async def pm_summary_node(state: PMState) -> dict[str, Any]:
    """Summary node: combine analysis result and PRD document into final output."""

    parts = []

    if state.analysis_result:
        parts.append("## 需求分析\n\n")
        parts.append(state.analysis_result)
        parts.append("\n\n---\n\n")

    if state.prd_document:
        parts.append("## PRD 文档\n\n")
        parts.append(state.prd_document)

    final_result = "".join(parts) if parts else "未能生成完整的PRD文档。"

    if state.clarification_rounds:
        summary_parts = ["\n\n---\n\n## 澄清记录\n"]
        for round_data in state.clarification_rounds:
            summary_parts.append(f"\n### 第{round_data.round_number}轮\n")
            for i, (q, a) in enumerate(zip(round_data.questions, round_data.answers), 1):
                summary_parts.append(f"**Q{i}:** {q}\n")
                summary_parts.append(f"**A{i}:** {a}\n\n")
        final_result += "".join(summary_parts)

    return {
        "final_result": final_result,
        "current_step": "end",
        "prd_confirmed": True,
        "messages": state.messages + [
            {"role": "assistant", "content": "PRD文档生成完成。"},
        ],
    }