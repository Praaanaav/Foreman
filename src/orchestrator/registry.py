import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from orchestrator.schemas import AgentType

log = logging.getLogger("orchestrator.tools")


class ToolCallRecord(BaseModel):
    tool: str
    agent: str
    inputs: dict
    output: str | None = None
    success: bool
    error: str | None = None
    latency_ms: float


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    input_model: type[BaseModel]
    allowed_agents: set[AgentType]
    max_calls_per_minute: int
    _recent_calls: deque = field(default_factory=deque)

    def check_rate_limit(self) -> None:
        now = time.monotonic()
        while self._recent_calls and now - self._recent_calls[0] > 60:
            self._recent_calls.popleft()
        if len(self._recent_calls) >= self.max_calls_per_minute:
            raise RuntimeError(f"Rate limit reached for '{self.name}'")
        self._recent_calls.append(now)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.call_log: list[ToolCallRecord] = []

    def register(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        allowed_agents: list[AgentType],
        max_calls_per_minute: int = 30,
    ):
        def decorator(func: Callable[..., str]):
            self._tools[name] = Tool(
                name=name,
                description=description,
                func=func,
                input_model=input_model,
                allowed_agents=set(allowed_agents),
                max_calls_per_minute=max_calls_per_minute,
            )
            return func

        return decorator

    def tools_for(self, agent: AgentType) -> list[Tool]:
        return [t for t in self._tools.values() if agent in t.allowed_agents]

    def call(self, name: str, agent: AgentType, **kwargs) -> ToolCallRecord:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        if agent not in tool.allowed_agents:
            raise PermissionError(f"'{agent.value}' agent may not use '{name}'")

        start = time.perf_counter()
        output, error = None, None
        try:
            tool.check_rate_limit()
            args = tool.input_model(**kwargs)  # validates the inputs
            output = tool.func(**args.model_dump())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        record = ToolCallRecord(
            tool=name,
            agent=agent.value,
            inputs=kwargs,
            output=output,
            success=error is None,
            error=error,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        self.call_log.append(record)
        log.info("tool_call %s", record.model_dump_json())
        return record


registry = ToolRegistry()