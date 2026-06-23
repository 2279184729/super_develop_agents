"""MetaGPT-style Role abstraction for agent development.

Inspired by MetaGPT's Role pattern:
- Each Role has a profile, goal, and constraints
- Role._think() → analyze context and decide next action
- Role._act() → execute the chosen action
- Roles communicate through a shared message pool

This provides a consistent pattern for building all agent roles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoleContext:
    """Rich context for a role — inspired by MetaGPT's RoleContext."""

    name: str = ""
    profile: str = ""  # e.g. "资深产品经理，15年经验"
    goal: str = ""  # e.g. "生成高质量PRD文档"
    constraints: str = ""  # e.g. "必须包含验收标准"
    watch_actions: list[str] = field(default_factory=list)  # 关注的 Action 类型


@dataclass
class Action:
    """A typed action — inspired by MetaGPT's Action class."""

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)


class Role(ABC):
    """Base Role class — inspired by MetaGPT's Role pattern.

    Subclasses implement:
    - _init_actions(): define available actions
    - _think(): decide what to do next
    - _act(): execute the chosen action
    """

    def __init__(self, context: RoleContext):
        self.context = context
        self.actions: list[Action] = []
        self._memory: list[dict] = []  # message pool
        self._init_actions()

    @abstractmethod
    def _init_actions(self) -> None:
        """Initialize role-specific actions."""

    @abstractmethod
    async def _think(self) -> Action | None:
        """Observe current state and decide next action.

        Returns the action to execute, or None if no action needed.
        """

    @abstractmethod
    async def _act(self, action: Action) -> Any:
        """Execute the chosen action and return result."""

    def _observe(self, message: dict) -> None:
        """Receive a message from the shared message pool."""
        self._memory.append(message)

    async def run(self) -> Any:
        """Main loop: think → act — inspired by MetaGPT's _react()."""
        action = await self._think()
        if action is None:
            return None
        return await self._act(action)


# ═══════════════════════════════════════════════════════════
#  PM Role Implementations
# ═══════════════════════════════════════════════════════════

class PMRoleContext(RoleContext):
    """Pre-configured context for PM roles."""

    product_domain: str = ""
    target_users: str = ""
    tech_stack: str = ""


def create_pm_role(role_type: str, **kwargs) -> Role:
    """Factory for creating PM roles — inspired by MetaGPT's role factory."""
    contexts = {
        "clarifier": RoleContext(
            name="需求澄清师",
            profile="资深产品经理，15年互联网产品经验，擅长通过精准提问挖掘用户真实需求",
            goal="通过结构化提问澄清模糊需求，确保需求完整度达到80%以上",
            constraints="每轮最多3个问题，问题要具体有引导性，提供选项参考",
            watch_actions=["UserQuery", "ClarificationAnswer"],
        ),
        "analyzer": RoleContext(
            name="需求分析师",
            profile="资深需求分析专家，擅长将模糊需求转化为结构化分析报告",
            goal="生成包含用户画像、功能优先级、风险识别的结构化分析报告",
            constraints="必须包含P0/P1/P2优先级、用户故事、验收标准",
            watch_actions=["ClarificationComplete", "UserQuery"],
        ),
        "prd_writer": RoleContext(
            name="PRD撰写师",
            profile="资深产品文档专家，15年PRD撰写经验，输出可直接交付开发的PRD",
            goal="生成完整、专业、可执行的PRD文档",
            constraints="功能描述必须具体到开发者可直接编码，验收标准必须可测试可量化",
            watch_actions=["AnalysisComplete"],
        ),
    }
    return contexts.get(role_type, RoleContext(**kwargs))


# ═══════════════════════════════════════════════════════════
#  Chaos Testing Role Implementations
# ═══════════════════════════════════════════════════════════

class ChaosRoleContext(RoleContext):
    """Pre-configured context for chaos testing roles."""

    target_agent: str = "pm"
    scenarios: list[str] = field(default_factory=list)


def create_chaos_role(role_type: str, **kwargs) -> RoleContext:
    """Factory for creating chaos testing roles."""
    contexts = {
        "injector": ChaosRoleContext(
            name="混沌注入器",
            profile="混沌工程专家，精通故障注入和异常场景构建",
            goal="向目标Agent注入真实世界的噪声和异常，测试其鲁棒性",
            constraints="注入后必须保持核心意图可识别，不能完全破坏输入语义",
        ),
        "evaluator": ChaosRoleContext(
            name="质量评估师",
            profile="资深测试架构师，精通多维度质量评估方法论",
            goal="从硬规则和语义两个维度全面评估Agent响应质量",
            constraints="评估必须基于可量化的指标，每个指标必须有明确的评分理由",
        ),
        "reporter": ChaosRoleContext(
            name="报告生成器",
            profile="测试报告专家，擅长将测试数据转化为结构化洞察",
            goal="生成包含通过率、严重度分布、失败原因分析的结构化报告",
            constraints="报告必须包含可操作的建议，而非仅罗列数据",
        ),
    }
    return contexts.get(role_type, ChaosRoleContext(**kwargs))