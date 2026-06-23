"""AI-TestPilot agent — test case generation, script generation, defect analysis.

Inspired by cover-agent's iterative coverage-guided pattern:
1. Analyze source → identify gaps
2. Generate tests → validate
3. Measure quality → refine
4. Repeat until quality threshold met
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# ═══════════════════════════════════════════════════════════
#  Cover-agent-style data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class GenerationConfig:
    """Configuration for test generation — inspired by cover-agent's config."""

    target_modules: list[str] = field(default_factory=list)
    priority_distribution: dict = field(default_factory=lambda: {"P0": 20, "P1": 30, "P2": 30, "P3": 20})
    max_cases: int = 30
    min_steps_per_case: int = 3
    quality_threshold: float = 0.7
    iteration_limit: int = 3


@dataclass
class GenerationQuality:
    """Quality assessment of generated test cases."""

    total_cases: int = 0
    has_all_fields: bool = False
    has_enough_steps: bool = False
    priority_distribution_ok: bool = False
    coverage_score: float = 0.0
    issues: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        return self.coverage_score >= 0.7 and self.has_all_fields and self.has_enough_steps


def _validate_case_quality(cases: list[dict], config: GenerationConfig) -> GenerationQuality:
    """Validate generated test cases against quality criteria — cover-agent style."""
    quality = GenerationQuality()
    quality.total_cases = len(cases)

    if not cases:
        quality.issues.append("未生成任何用例")
        return quality

    required_fields = {"id", "module", "title", "precondition", "steps", "expected", "priority"}
    quality.has_all_fields = all(
        required_fields.issubset(case.keys()) for case in cases
    )
    if not quality.has_all_fields:
        quality.issues.append("部分用例缺少必需字段")

    quality.has_enough_steps = all(
        len(case.get("steps", [])) >= config.min_steps_per_case
        for case in cases
    )
    if not quality.has_enough_steps:
        quality.issues.append(f"部分用例操作步骤不足{config.min_steps_per_case}步")

    if cases:
        priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for case in cases:
            p = case.get("priority", "P2")
            if p in priority_counts:
                priority_counts[p] += 1
        total = len(cases)
        expected_p0 = config.priority_distribution.get("P0", 20) / 100 * total
        quality.priority_distribution_ok = (priority_counts.get("P0", 0) >= expected_p0 * 0.5)
        if not quality.priority_distribution_ok:
            quality.issues.append("P0用例数量不足")

    score = 0.0
    if quality.has_all_fields: score += 0.35
    if quality.has_enough_steps: score += 0.35
    if quality.priority_distribution_ok: score += 0.3
    quality.coverage_score = round(score, 2)

    return quality


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
    document_text: str, extra_context: str = "", config: GenerationConfig | None = None,
) -> AsyncIterator[dict]:
    """Generate test cases with iterative refinement — cover-agent style.

    1. First pass: generate initial cases
    2. Validate quality (field completeness, step count, priority distribution)
    3. If quality < threshold, refine with targeted prompts
    4. Repeat until quality acceptable or iteration limit reached
    """
    cfg = config or GenerationConfig()
    yield {"event": "status", "data": json.dumps({"label": "正在分析文档内容...", "progress": 15})}

    context = document_text[:8000]
    if extra_context:
        context = f"额外测试重点:\n{extra_context}\n\n文档内容:\n{context}"

    llm = _get_llm(temperature=0.3)
    all_cases: list[dict] = []

    for iteration in range(cfg.iteration_limit):
        progress = 20 + (iteration * 25)
        label = f"正在生成测试用例 (第{iteration + 1}轮)..." if iteration > 0 else "正在生成测试用例..."
        yield {"event": "status", "data": json.dumps({"label": label, "progress": progress})}

        # Build refinement prompt if we have previous results
        if iteration > 0 and all_cases:
            quality = _validate_case_quality(all_cases, cfg)
            if quality.is_acceptable:
                yield {"event": "status", "data": json.dumps({"label": "质量达标，停止迭代", "progress": 90})}
                break
            refinement = f"上一轮生成的问题是: {', '.join(quality.issues)}。请改进："
        else:
            refinement = ""

        messages = [
            SystemMessage(content=CASE_GENERATION_PROMPT),
            HumanMessage(content=f"{refinement}请基于以下文档生成测试用例:\n\n{context}"),
        ]

        try:
            response = llm.invoke(messages)
            text = _extract_text(response.content)
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

            if cases:
                all_cases = cases

            yield {"event": "status", "data": json.dumps({
                "label": f"第{iteration + 1}轮生成 {len(cases)} 条用例",
                "progress": progress + 15,
            })}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            break

    # Final quality assessment
    quality = _validate_case_quality(all_cases, cfg)
    yield {"event": "status", "data": json.dumps({"label": f"质量评估: {quality.coverage_score:.0%} | 共{len(all_cases)}条", "progress": 95})}

    yield {
        "event": "result",
        "data": json.dumps({
            "cases": all_cases,
            "count": len(all_cases),
            "quality": {
                "coverage_score": quality.coverage_score,
                "is_acceptable": quality.is_acceptable,
                "issues": quality.issues,
            },
        }, ensure_ascii=False),
    }


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