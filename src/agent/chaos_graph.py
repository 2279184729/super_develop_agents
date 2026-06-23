"""Chaos testing LangGraph definition and execution functions.

Simple linear graph: orchestrator → executor → reporter → END
"""

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from src.storage.checkpoint import create_checkpointer, get_chaos_checkpoint_db_path
from langgraph.graph import END, StateGraph

from .chaos_state import ChaosState, TestCase, BaselineRules
from .chaos_workers import chaos_executor_node, chaos_reporter_node
from .sse_utils import sse, extract_output


def chaos_orchestrator_node(state: ChaosState) -> dict:
    """Parse user query into test configuration and dispatch to executor."""
    config = {}
    try:
        config = json.loads(state.user_query)
    except (json.JSONDecodeError, TypeError):
        pass

    if state.test_cases:
        return {"current_step": "executor"}

    if config:
        test_cases = []
        for i, case_data in enumerate(config.get("test_cases", [])):
            test_cases.append(TestCase(
                id=case_data.get("id", f"case_{i+1}"),
                input_text=case_data.get("input_text", ""),
                expected_behavior=case_data.get("expected_behavior", ""),
                scenario=case_data.get("scenario", "text_noise"),
            ))

        baseline = BaselineRules(
            max_response_time_ms=config.get("max_response_time_ms", 30000),
            tool_whitelist=config.get("tool_whitelist", []),
            blocked_outputs=config.get("blocked_outputs", []),
            business_rules=config.get("business_rules", ""),
        )

        return {
            "test_cases": test_cases,
            "chaos_scenarios": config.get("chaos_scenarios", ["text_noise"]),
            "baseline_rules": baseline,
            "target_agent": config.get("target_agent", "pm"),
            "external_config": config.get("external_config"),
            "current_step": "executor",
        }

    return {"current_step": "executor"}


def chaos_should_continue(state: dict[str, Any] | Any) -> str:
    """Route based on current_step."""
    if isinstance(state, dict):
        step = state.get("current_step", "end")
    else:
        step = getattr(state, "current_step", "end")
    if step == "executor":
        return "executor"
    elif step == "reporter":
        return "reporter"
    return END


def build_chaos_graph():
    """Build and compile the chaos testing graph."""
    workflow = StateGraph(ChaosState)

    workflow.add_node("orchestrator", chaos_orchestrator_node)
    workflow.add_node("executor", chaos_executor_node)
    workflow.add_node("reporter", chaos_reporter_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator", chaos_should_continue,
        {"executor": "executor", END: END},
    )
    workflow.add_conditional_edges(
        "executor", chaos_should_continue,
        {"reporter": "reporter", END: END},
    )
    workflow.add_conditional_edges(
        "reporter", chaos_should_continue,
        {END: END},
    )

    memory = create_checkpointer(get_chaos_checkpoint_db_path())
    app = workflow.compile(checkpointer=memory)
    return app


# Lazy graph singleton
import threading

_chaos_graph_lock = threading.Lock()
_chaos_graph_app = None


def _get_chaos_graph():
    global _chaos_graph_app
    if _chaos_graph_app is not None:
        return _chaos_graph_app
    with _chaos_graph_lock:
        if _chaos_graph_app is None:
            _chaos_graph_app = build_chaos_graph()
        return _chaos_graph_app


# ── Streaming execution ──

async def run_chaos_stream(
    config_data: dict,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run chaos test and stream SSE progress events."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    test_cases = []
    for i, case_data in enumerate(config_data.get("test_cases", [])):
        test_cases.append(TestCase(
            id=case_data.get("id", f"case_{i+1}"),
            input_text=case_data.get("input_text", ""),
            expected_behavior=case_data.get("expected_behavior", ""),
            scenario=case_data.get("scenario", "text_noise"),
        ))

    baseline = BaselineRules(
        max_response_time_ms=config_data.get("max_response_time_ms", 30000),
        tool_whitelist=config_data.get("tool_whitelist", []),
        blocked_outputs=config_data.get("blocked_outputs", []),
        business_rules=config_data.get("business_rules", ""),
    )

    initial_state = ChaosState(
        user_query=json.dumps(config_data, ensure_ascii=False),
        target_agent=config_data.get("target_agent", "pm"),
        external_config=config_data.get("external_config"),
        test_cases=test_cases,
        chaos_scenarios=config_data.get("chaos_scenarios", ["text_noise"]),
        baseline_rules=baseline,
        concurrency=config_data.get("concurrency", 3),
        timeout_per_case=config_data.get("timeout_per_case", 60),
        messages=[{"role": "human", "content": f"Run chaos test with {len(test_cases)} cases"}],
    )

    config = {"configurable": {"thread_id": thread_id}}

    yield sse("connected", {"thread_id": thread_id})
    yield sse("status", {"node": "start", "status": "started", "label": f"开始混沌测试，共 {len(test_cases)} 个用例"})

    try:
        async for event in _get_chaos_graph().astream_events(
            initial_state.model_dump(), config=config, version="v2",
        ):
            kind = event.get("event", "")
            node = event.get("name", "")

            if kind == "on_chain_start" and node == "executor":
                yield sse("status", {"node": "executor", "status": "started", "label": "执行混沌测试用例中..."})

            elif kind == "on_chain_end" and node == "executor":
                output = extract_output(event)
                if output and "test_results" in output:
                    for i, tr in enumerate(output["test_results"]):
                        tr_data = tr.model_dump() if hasattr(tr, "model_dump") else tr
                        yield sse("case_result", {
                            "index": i + 1,
                            "case_id": tr_data.get("case_id", ""),
                            "scenario": tr_data.get("scenario", ""),
                            "passed": tr_data.get("passed", False),
                            "severity": tr_data.get("severity", "low"),
                            "response_time_ms": tr_data.get("response_time_ms", 0),
                        })

            elif kind == "on_chain_start" and node == "reporter":
                yield sse("status", {"node": "reporter", "status": "started", "label": "生成测试报告..."})

            elif kind == "on_chain_end" and node == "reporter":
                output = extract_output(event)
                if output:
                    yield sse("done", {
                        "summary": output.get("summary"),
                        "final_result": output.get("final_result"),
                    })

    except Exception as e:
        yield sse("error", {"error": str(e)})


async def get_chaos_state(thread_id: str) -> dict | None:
    """Get chaos test state for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await _get_chaos_graph().aget_state(config)
    if state and state.values:
        return {**state.values, "__interrupted": bool(state.next)}
    return None


async def list_chaos_threads(limit: int = 30) -> list[dict]:
    """List recent chaos test threads."""
    from src.storage.checkpoint import list_thread_ids, get_chaos_checkpoint_db_path

    thread_ids = list_thread_ids(limit, db_path=get_chaos_checkpoint_db_path())
    threads = []
    for tid in thread_ids:
        try:
            state = await get_chaos_state(tid)
        except Exception:
            continue
        if state and state.get("user_query"):
            if not state.get("test_results") and not state.get("summary"):
                continue
            threads.append({
                "thread_id": tid,
                "title": f"混沌测试 ({len(state.get('test_results', []))} 用例)",
                "step": state.get("current_step", ""),
                "has_report": bool(state.get("summary")),
            })
        if len(threads) >= limit:
            break
    return threads