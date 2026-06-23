"""Dynamic Worker registry — plugin-style registration for agent workers."""

from collections.abc import Callable

from pydantic import BaseModel, Field


class WorkerConfig(BaseModel):
    """Worker 配置模型"""

    name: str = Field(description="Worker 唯一标识名称")
    description: str = Field(description="Worker 功能描述，用于编排者选择")
    node_fn: Callable = Field(description="Worker 执行函数 (async callable)")
    keywords: list[str] = Field(
        default_factory=list,
        description="触发关键词，用于自动匹配任务",
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM 模型覆盖（不填则使用默认模型）",
    )
    tools_filter: list[str] | None = Field(
        default=None,
        description="工具过滤（不填则使用全部 MCP 工具）",
    )

    model_config = {"arbitrary_types_allowed": True}


class WorkerRegistry:
    """插件式 Worker 注册表"""

    _workers: dict[str, WorkerConfig] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        keywords: list[str] | None = None,
        llm_model: str | None = None,
        tools_filter: list[str] | None = None,
    ) -> Callable:
        """装饰器方式注册 Worker"""

        def decorator(fn: Callable) -> Callable:
            config = WorkerConfig(
                name=name,
                description=description,
                node_fn=fn,
                keywords=keywords or [],
                llm_model=llm_model,
                tools_filter=tools_filter,
            )
            cls._workers[name] = config
            return fn

        return decorator

    @classmethod
    def register_worker(cls, config: WorkerConfig) -> None:
        """直接注册 Worker"""
        cls._workers[config.name] = config

    @classmethod
    def get_worker(cls, name: str) -> WorkerConfig | None:
        """获取 Worker 配置"""
        return cls._workers.get(name)

    @classmethod
    def get_all_workers(cls) -> dict[str, WorkerConfig]:
        """获取所有已注册的 Worker"""
        return dict(cls._workers)

    @classmethod
    def get_worker_descriptions(cls) -> str:
        """获取所有 Worker 的描述信息（用于编排者 prompt）"""
        lines = []
        for i, (name, config) in enumerate(cls._workers.items(), 1):
            keywords = ", ".join(config.keywords) if config.keywords else "通用"
            lines.append(f"{i}. {name}: {config.description} [关键词: {keywords}]")
        return "\n".join(lines)

    @classmethod
    def match_worker(cls, task_description: str) -> str | None:
        """根据任务描述匹配最合适的 Worker"""
        task_lower = task_description.lower()
        for name, config in cls._workers.items():
            for keyword in config.keywords:
                if keyword in task_lower:
                    return name
        if cls._workers:
            return next(iter(cls._workers))
        return None

    @classmethod
    def clear(cls) -> None:
        """清空注册表（主要用于测试）"""
        cls._workers.clear()