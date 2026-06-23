"""Chaos testing agent state definitions."""

from typing import Any
from pydantic import BaseModel, Field
from .state import AgentState


class TestCase(BaseModel):
    """A single test case for chaos testing."""
    id: str = Field(default="", description="Test case identifier")
    input_text: str = Field(default="", description="Original input to the agent")
    expected_behavior: str = Field(default="", description="Expected behavior description")
    scenario: str = Field(default="text_noise", description="Chaos scenario type")


class BaselineRules(BaseModel):
    """Steady-state baseline configuration."""
    max_response_time_ms: int = Field(default=30000, description="Max allowed response time in ms")
    tool_whitelist: list[str] = Field(default_factory=list, description="Allowed tool names")
    blocked_outputs: list[str] = Field(default_factory=list, description="Blocked output keywords/patterns")
    business_rules: str = Field(default="", description="Business rules description for LLM judge")


class TestResult(BaseModel):
    """Result of a single chaos test case."""
    case_id: str
    scenario: str
    original_input: str
    modified_input: str
    agent_response: str
    response_time_ms: int = 0
    hard_rule_results: dict = Field(default_factory=dict)
    llm_judge_scores: dict = Field(default_factory=dict)
    severity: str = "low"
    passed: bool = False
    error: str | None = None


class ChaosState(AgentState):
    """Chaos testing agent state — extends AgentState for graph compatibility."""

    target_agent: str = Field(default="pm", description="Target: pm / external")
    external_config: dict | None = Field(default=None, description="External agent API config")
    test_cases: list[TestCase] = Field(default_factory=list)
    chaos_scenarios: list[str] = Field(default_factory=list)
    baseline_rules: BaselineRules = Field(default_factory=BaselineRules)
    concurrency: int = Field(default=3)
    timeout_per_case: int = Field(default=60)
    test_results: list[TestResult] = Field(default_factory=list)
    summary: dict | None = Field(default=None)
    running: bool = Field(default=False)