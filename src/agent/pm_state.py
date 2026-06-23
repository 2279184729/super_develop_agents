"""PM Agent state definition - extends AgentState with clarification and PRD fields."""

from typing import List, Optional
from pydantic import BaseModel, Field
from .state import AgentState


class ClarificationRound(BaseModel):
    """A single round of clarification Q&A."""

    round_number: int = Field(..., description="轮次编号")
    questions: List[str] = Field(default_factory=list, description="本轮提出的问题")
    answers: List[str] = Field(default_factory=list, description="用户的回答")


class PMState(AgentState):
    """PM Agent state - extends AgentState with clarification loop and PRD fields."""

    clarification_rounds: List[ClarificationRound] = Field(
        default_factory=list,
        description="已完成的澄清轮次历史"
    )
    current_questions: List[str] = Field(
        default_factory=list,
        description="当前轮次的澄清问题"
    )
    current_answers: List[str] = Field(
        default_factory=list,
        description="用户对当前问题的回答"
    )
    max_clarification_rounds: int = Field(
        default=3,
        description="最大澄清轮次数"
    )

    pm_decision: str = Field(
        default="need_clarification",
        description="编排者决策: need_clarification | ready_to_generate"
    )
    information_sufficiency: float = Field(
        default=0.0,
        description="信息充足度 0.0-1.0"
    )

    analysis_result: Optional[str] = Field(
        default=None,
        description="需求分析结果"
    )
    prd_document: Optional[str] = Field(
        default=None,
        description="生成的 PRD 文档 (Markdown)"
    )
    prd_confirmed: bool = Field(
        default=False,
        description="用户是否确认了 PRD"
    )
    prd_feedback: Optional[str] = Field(
        default=None,
        description="用户对 PRD 的修改意见"
    )