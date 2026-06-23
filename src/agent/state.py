"""Agent state definitions for the PM + Chaos system."""

from typing import Any

from pydantic import BaseModel, Field


class WorkerResult(BaseModel):
    """单个Worker的执行结果"""
    worker_name: str
    status: str = Field(description="执行状态: success/error")
    result: str = Field(description="Worker的输出结果")
    error: str | None = Field(default=None, description="错误信息（如果有）")
    tools_used: list[str] = Field(default_factory=list, description="使用过的工具")


class AgentState(BaseModel):
    """多Agent系统的全局状态"""

    user_query: str = Field(description="用户的原始查询")

    task_plan: list[str] = Field(default_factory=list, description="编排者拆解的任务列表")

    pending_workers: list[str] = Field(
        default_factory=list,
        description="待执行的Worker名称列表",
    )

    worker_results: dict[str, WorkerResult] = Field(
        default_factory=dict,
        description="各个Worker的执行结果",
    )

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="对话消息历史",
    )

    current_step: str = Field(default="orchestrator", description="当前执行的步骤")

    completed_tasks: list[str] = Field(
        default_factory=list,
        description="已完成的任务",
    )

    final_result: str | None = Field(default=None, description="最终的汇总结果")

    pending_approval: bool = Field(
        default=False,
        description="是否等待用户审批任务计划",
    )
    approval_feedback: str | None = Field(
        default=None,
        description="用户对任务计划的修改意见",
    )

    metadata: dict[str, Any] = Field(default_factory=dict, description="额外的元数据")