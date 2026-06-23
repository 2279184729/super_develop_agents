"""AI-TestPilot agent — test case generation, script generation, defect analysis."""

import json
import os
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def _get_llm(temperature: float = 0.3) -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
        temperature=temperature,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
    )


def _extract_text(content: Any) -> str:
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


CASE_GENERATION_PROMPT = """你是一位资深测试架构师，拥有15年软件测试经验。根据提供的文档内容，生成结构化的测试用例。

要求：
1. 每个用例必须包含：用例ID、所属模块、用例标题、前置条件、操作步骤（至少3步）、预期结果、优先级（P0/P1/P2/P3）
2. 覆盖正常流程、异常流程、边界条件
3. P0覆盖核心业务流程，P1覆盖重要分支，P2覆盖边缘场景，P3覆盖兼容性/性能
4. 用例标题简洁明确，操作步骤具体可执行
5. 优先级分布：P0约占20%，P1约占30%，P2约占30%，P3约占20%

输出格式（严格JSON数组）：
[
  {
    "id": "TC_001",
    "module": "模块名称",
    "title": "用例标题",
    "precondition": "前置条件",
    "steps": ["步骤1", "步骤2", "步骤3"],
    "expected": "预期结果",
    "priority": "P0"
  }
]

只输出JSON数组，不要包含其他文字。"""


async def generate_test_cases_stream(
    document_text: str, extra_context: str = ""
) -> AsyncIterator[dict]:
    """Generate test cases from document text with SSE streaming."""
    yield {"event": "status", "data": json.dumps({"label": "正在分析文档内容...", "progress": 20})}

    context = document_text[:8000]
    if extra_context:
        context = f"额外测试重点:\n{extra_context}\n\n文档内容:\n{context}"

    yield {"event": "status", "data": json.dumps({"label": "正在调用AI生成测试用例...", "progress": 40})}

    llm = _get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=CASE_GENERATION_PROMPT),
        HumanMessage(content=f"请基于以下文档生成测试用例:\n\n{context}"),
    ]

    try:
        response = llm.invoke(messages)
        text = _extract_text(response.content)

        yield {"event": "status", "data": json.dumps({"label": "正在解析生成结果...", "progress": 80})}

        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        cases = json.loads(text)
        if not isinstance(cases, list):
            cases = []

        yield {
            "event": "result",
            "data": json.dumps({"cases": cases, "count": len(cases)}, ensure_ascii=False),
        }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


SCRIPT_GENERATION_PROMPT = """你是一位资深测试开发工程师。根据测试用例生成自动化测试脚本。

{script_type_instruction}

输出格式：只输出完整的代码，不要包含解释文字。代码要可以直接运行。"""


async def generate_scripts_stream(
    cases: list[dict], script_type: str = "pytest"
) -> AsyncIterator[dict]:
    """Generate automation scripts from test cases."""
    yield {"event": "status", "data": json.dumps({"label": f"正在生成{script_type}脚本...", "progress": 30})}

    if script_type == "playwright":
        instruction = """使用Playwright + TypeScript生成UI自动化测试脚本。
要求：
- 使用@playwright/test框架
- 每个测试用例对应一个test()函数
- 包含适当的断言（expect）
- 添加合理的等待和超时处理"""
    else:
        instruction = """使用Pytest + requests生成接口自动化测试脚本。
要求：
- 使用pytest框架
- 每个测试用例对应一个test_函数
- 包含适当的assert断言
- 添加 fixture 和 conftest 配置"""

    prompt = SCRIPT_GENERATION_PROMPT.format(script_type_instruction=instruction)

    llm = _get_llm(temperature=0.2)
    cases_text = json.dumps(cases[:5], ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"请基于以下测试用例生成自动化脚本:\n\n{cases_text}"),
    ]

    try:
        response = llm.invoke(messages)
        script = _extract_text(response.content)
        yield {
            "event": "result",
            "data": json.dumps({"script": script, "script_type": script_type}, ensure_ascii=False),
        }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


DEFECT_ANALYSIS_PROMPT = """你是一位资深Debug专家，精通多种编程语言和调试技术。分析以下错误日志，提供根因分析和修复建议。

要求输出Markdown格式：

## 错误概述
[简要说明错误类型和影响]

## 根因分析
[详细分析错误的根本原因]

## 修复建议
[具体的修复方案，包含代码Diff]

### 修改文件
- **文件路径**: [精确的文件路径]
- **行号**: [精确的行号范围]

### 代码变更
```diff
[具体的Diff格式代码变更]
```

## 预防措施
[如何避免类似问题]"""


async def analyze_defect_stream(
    error_log: str, code_context: str = ""
) -> AsyncIterator[dict]:
    """Analyze defect from error log with LLM."""
    yield {"event": "status", "data": json.dumps({"label": "正在分析错误堆栈...", "progress": 30})}

    llm = _get_llm(temperature=0.2)
    prompt = f"错误日志:\n```\n{error_log[:5000]}\n```"
    if code_context:
        prompt += f"\n\n相关代码:\n```\n{code_context[:3000]}\n```"

    messages = [
        SystemMessage(content=DEFECT_ANALYSIS_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        analysis = _extract_text(response.content)
        yield {
            "event": "result",
            "data": json.dumps({"analysis": analysis}, ensure_ascii=False),
        }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}