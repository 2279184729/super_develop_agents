"""PM Agent LangGraph definition and execution functions.

Graph topology:
  orchestrator ──[need_clarification]──→ clarifier ──interrupt()──→ PAUSE
       ↑                                         │
       │                              user answers (resume)
       └──────────────────────────────────────────┘

  orchestrator ──[ready_to_generate]──→ analyzer → prd_generator → summary → END
"""

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from src.storage.checkpoint import create_checkpointer, get_pm_checkpoint_db_path
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .pm_state import PMState
from .pm_orchestrator import pm_orchestrator_node, pm_summary_node
from .pm_workers import clarifier_worker_node, analyzer_worker_node, prd_generator_worker_node
from .sse_utils import sse, extract_output, extract_stream_token

# Import workers to trigger registration
from . import pm_workers  # noqa: F401


# ═══════════════════════════════════════════════════════════
#  Routing function
# ═══════════════════════════════════════════════════════════

def pm_should_continue(state: dict[str, Any] | Any) -> str:
    """Route based on current_step and pm_decision."""
    if isinstance(state, dict):
        step = state.get("current_step", "end")
        decision = state.get("pm_decision", "need_clarification")
    else:
        step = getattr(state, "current_step", "end")
        decision = getattr(state, "pm_decision", "need_clarification")

    if step == "clarifier":
        return "clarifier"
    elif step == "analyzer":
        return "analyzer"
    elif step == "end":
        return END

    if decision == "ready_to_generate":
        return "analyzer"
    return "clarifier"


# ═══════════════════════════════════════════════════════════
#  Build PM Graph
# ═══════════════════════════════════════════════════════════

def build_pm_graph():
    """Build and compile the PM Agent graph."""
    workflow = StateGraph(PMState)

    workflow.add_node("orchestrator", pm_orchestrator_node)
    workflow.add_node("clarifier", clarifier_worker_node)
    workflow.add_node("analyzer", analyzer_worker_node)
    workflow.add_node("prd_generator", prd_generator_worker_node)
    workflow.add_node("summary", pm_summary_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        pm_should_continue,
        {"clarifier": "clarifier", "analyzer": "analyzer", END: END},
    )

    workflow.add_edge("clarifier", "orchestrator")
    workflow.add_edge("analyzer", "prd_generator")
    workflow.add_edge("prd_generator", "summary")

    workflow.add_conditional_edges("summary", pm_should_continue, {END: END})

    memory = create_checkpointer(get_pm_checkpoint_db_path())
    app = workflow.compile(checkpointer=memory)

    return app


# Lazy graph singleton
import threading

_pm_graph_lock = threading.Lock()
_pm_graph_app = None


def _get_pm_graph():
    """Return the compiled PM graph, creating it lazily on first call."""
    global _pm_graph_app
    if _pm_graph_app is not None:
        return _pm_graph_app
    with _pm_graph_lock:
        if _pm_graph_app is None:
            _pm_graph_app = build_pm_graph()
        return _pm_graph_app


# ═══════════════════════════════════════════════════════════
#  Stream processing helper
# ═══════════════════════════════════════════════════════════

async def _process_pm_events(
    graph,
    input_data: Any,
    config: dict,
) -> AsyncGenerator[str, None]:
    """Process astream_events and yield SSE event strings."""
    async for event in graph.astream_events(input_data, config=config, version="v2"):
        kind = event.get("event", "")
        node = event.get("name", "")

        if kind == "on_chain_start" and node in ("orchestrator", "clarifier", "analyzer", "prd_generator", "summary"):
            labels = {
                "orchestrator": "评估需求中...",
                "clarifier": "生成澄清问题...",
                "analyzer": "分析需求中...",
                "prd_generator": "生成PRD文档...",
                "summary": "汇总结果中...",
            }
            yield sse("status", {"node": node, "status": "started", "label": labels.get(node, node)})

        elif kind == "on_chain_end":
            output = extract_output(event)
            if output:
                if node == "orchestrator":
                    decision = output.get("pm_decision")
                    sufficiency = output.get("information_sufficiency")
                    yield sse("orchestrator_decision", {
                        "decision": decision,
                        "information_sufficiency": sufficiency,
                    })

                elif node == "analyzer":
                    analysis = output.get("analysis_result")
                    if analysis:
                        yield sse("analysis_done", {"analysis_result": analysis})

                elif node == "prd_generator":
                    prd = output.get("prd_document")
                    if prd:
                        yield sse("prd_ready", {"prd_document": prd})

                elif node == "summary":
                    final = output.get("final_result")
                    prd = output.get("prd_document")
                    if final:
                        yield sse("done", {
                            "final_result": final,
                            "prd_document": prd,
                        })

        elif kind == "on_chat_model_stream":
            token_text = extract_stream_token(event)
            if token_text:
                yield sse("token", {"content": token_text})


# ═══════════════════════════════════════════════════════════
#  Execution functions
# ═══════════════════════════════════════════════════════════

async def run_pm_stream(
    user_query: str,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Start or continue PM Agent conversation. Yields SSE events."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    existing_state = await _get_pm_graph().aget_state(config)

    if existing_state and existing_state.values:
        prior = existing_state.values
        prior_messages = prior.get("messages", [])
        if not isinstance(prior_messages, list):
            prior_messages = []
        input_data = {
            **prior,
            "user_query": user_query,
            "messages": prior_messages + [{"role": "user", "content": user_query}],
            "current_step": "orchestrator",
        }
    else:
        input_data = PMState(
            user_query=user_query,
            messages=[{"role": "user", "content": user_query}],
        ).model_dump()

    yield sse("connected", {"thread_id": thread_id})

    try:
        async for event in _process_pm_events(_get_pm_graph(), input_data, config):
            yield event
    except Exception as e:
        yield sse("error", {"error": str(e)})
        return

    state = await _get_pm_graph().aget_state(config)
    if state and state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    if isinstance(intr.value, dict) and intr.value.get("type") == "clarification_required":
                        yield sse("clarification_required", {
                            "questions": intr.value.get("questions", []),
                            "round_number": intr.value.get("round_number", 1),
                        })
                        return

    yield sse("status", {"node": "completed", "status": "finished"})


async def answer_pm_questions(
    thread_id: str,
    answers: list[str],
) -> AsyncGenerator[str, None]:
    """Resume PM graph with user answers to clarification questions."""
    config = {"configurable": {"thread_id": thread_id}}

    yield sse("status", {"node": "resumed", "answers_count": len(answers)})

    try:
        async for event in _process_pm_events(
            _get_pm_graph(),
            Command(resume=answers),
            config,
        ):
            yield event
    except Exception as e:
        yield sse("error", {"error": str(e)})
        return

    state = await _get_pm_graph().aget_state(config)
    if state and state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    if isinstance(intr.value, dict) and intr.value.get("type") == "clarification_required":
                        yield sse("clarification_required", {
                            "questions": intr.value.get("questions", []),
                            "round_number": intr.value.get("round_number", 1),
                        })
                        return


async def confirm_prd(
    thread_id: str,
    approve: bool = True,
    feedback: str | None = None,
) -> AsyncGenerator[str, None]:
    """Confirm or request revision of the generated PRD."""
    config = {"configurable": {"thread_id": thread_id}}

    if not approve:
        await _get_pm_graph().aupdate_state(config, {
            "prd_feedback": "用户要求修改PRD",
            "prd_confirmed": False,
        }, as_node="analyzer")
    elif feedback:
        await _get_pm_graph().aupdate_state(config, {
            "prd_feedback": feedback,
            "prd_confirmed": False,
        }, as_node="analyzer")
    else:
        await _get_pm_graph().aupdate_state(config, {
            "prd_confirmed": True,
        }, as_node="summary")

    yield sse("status", {"node": "prd_confirmed", "approved": approve})

    try:
        async for event in _process_pm_events(_get_pm_graph(), None, config):
            yield event
    except Exception as e:
        yield sse("error", {"error": str(e)})


async def get_pm_graph_state(thread_id: str) -> dict[str, Any] | None:
    """Get current PM graph state, including interrupt data if paused."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await _get_pm_graph().aget_state(config)
    if state and state.values:
        result = {**state.values, "__interrupted": bool(state.next)}
        if state.next and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    for intr in task.interrupts:
                        if isinstance(intr.value, dict) and intr.value.get("type") == "clarification_required":
                            result["pending_clarification"] = {
                                "questions": intr.value.get("questions", []),
                                "round_number": intr.value.get("round_number", 1),
                            }
        return result
    return None


async def list_pm_threads(limit: int = 50) -> list[dict]:
    """List recent PM Agent conversation threads."""
    from src.storage.checkpoint import list_thread_ids, get_pm_checkpoint_db_path

    thread_ids = list_thread_ids(limit, db_path=get_pm_checkpoint_db_path())
    threads = []
    for tid in thread_ids:
        try:
            state = await get_pm_graph_state(tid)
        except Exception:
            continue
        if state and state.get("user_query"):
            if not state.get("pm_decision") and not state.get("prd_document"):
                continue
            threads.append({
                "thread_id": tid,
                "title": state["user_query"][:60] + ("..." if len(state["user_query"]) > 60 else ""),
                "step": state.get("current_step", ""),
                "has_prd": bool(state.get("prd_document")),
            })
        if len(threads) >= limit:
            break
    return threads