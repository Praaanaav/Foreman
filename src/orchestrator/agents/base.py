"""Shared execution primitives for specialist agents.

A specialist runs a bounded ReAct-style tool loop:
  LLM -> optional tool calls (through the ToolRegistry) -> observe -> ...
  -> final structured AgentOutput.

Everything is traced so the run is inspectable after the fact.
"""
from __future__ import annotations

import datetime
import json
import logging
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from orchestrator import config, llm, registry
from orchestrator.schemas import AgentOutput, AgentType, TraceEvent

log = logging.getLogger("orchestrator.agents")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def llm_event(agent: str, detail: dict, latency_ms: float, usage: dict) -> dict:
    return TraceEvent(
        agent=agent,
        event="llm_call",
        detail={**detail, **usage},
        latency_ms=round(latency_ms, 1),
        timestamp=_now(),
    ).model_dump()


def tool_event(agent: str, record) -> dict:
    return TraceEvent(
        agent=agent,
        event="tool_call",
        detail={
            "tool": record.tool,
            "inputs": record.inputs,
            "success": record.success,
            "error": record.error,
        },
        latency_ms=record.latency_ms,
        timestamp=_now(),
    ).model_dump()


def decision_event(agent: str, detail: dict) -> dict:
    return TraceEvent(
        agent=agent, event="decision", detail=detail, timestamp=_now()
    ).model_dump()


def run_tool_loop(
    agent: AgentType,
    system: str,
    brief: str,
    model=None,
    max_iters: int | None = None,
) -> tuple[dict, list[dict]]:
    """Run one specialist's bounded tool loop.

    Returns (AgentOutput as dict, trace events for this run).
    """
    model = model or llm.get_model("worker")
    tools = registry.registry.to_langchain_tools(agent)
    max_iters = max_iters or config.MAX_TOOL_ITERS

    messages = [SystemMessage(content=system), HumanMessage(content=brief)]
    events: list[dict] = []
    final_text = ""
    response: AIMessage | None = None

    for i in range(max_iters):
        started = time.perf_counter()
        response = model.invoke(
            messages, tools=tools if i < max_iters - 1 else None
        )
        latency = (time.perf_counter() - started) * 1000
        events.append(
            llm_event(
                agent.value,
                {"iteration": i, "model": getattr(model, "model", None)},
                latency,
                llm.usage_of(response),
            )
        )
        messages.append(response)

        if not response.tool_calls:
            final_text = response.content or ""
            break

        for tool_call in response.tool_calls:
            record = registry.registry.call(
                tool_call["name"], agent, **(tool_call.get("args") or {})
            )
            events.append(tool_event(agent, record))
            messages.append(
                ToolMessage(
                    content=record.output or f"TOOL ERROR: {record.error}",
                    tool_call_id=tool_call["id"],
                )
            )

    if not final_text:
        final_text = (
            (response.content if response and response.content else "")
            or "Research incomplete: no final answer within the tool budget."
        )

    return parse_agent_output(final_text), events


def parse_agent_output(text: str) -> AgentOutput:
    try:
        return llm.parse_json(text, AgentOutput)
    except Exception:
        # Keep the run moving: use the raw text and flag low confidence.
        return AgentOutput(
            content=text,
            confidence=0.5,
            needs_attention=True,
            attention_reason="Structured output could not be parsed; using raw agent text.",
        )


def with_retries(fn, agent: AgentType, fallback: dict) -> tuple[dict, list[dict]]:
    """Run fn() with config.LLM_RETRIES retries on LLM errors.

    Returns (result, retry trace events). On final failure returns the
    fallback so the pipeline can continue and the reviewer can see it.
    """
    events: list[dict] = []
    for attempt in range(config.LLM_RETRIES + 1):
        try:
            return fn(), events
        except Exception as exc:
            log.warning(
                "agent %s attempt %d/%d failed: %s",
                agent.value,
                attempt + 1,
                config.LLM_RETRIES + 1,
                exc,
            )
            events.append(
                TraceEvent(
                    agent=agent.value,
                    event="retry",
                    detail={
                        "attempt": attempt + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    timestamp=_now(),
                ).model_dump()
            )
    return fallback, events


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def research_brief(state: dict, focus: str) -> str:
    """Build the shared context block every research specialist receives."""
    plan = state.get("plan") or {}
    requirements = plan.get("requirements") or {}
    lines = [
        "ORIGINAL REQUEST:",
        state.get("request", ""),
        "",
        "PARSED REQUIREMENTS:",
        json.dumps(requirements, ensure_ascii=False, indent=2),
    ]
    profile = state.get("profile") or {}
    if profile:
        lines += ["", "TRAVELER PROFILE:", json.dumps(profile, ensure_ascii=False)]
    memories = state.get("memories") or []
    if memories:
        lines += ["", "RELEVANT TRAVELER MEMORIES:", *(f"- {m}" for m in memories)]
    errors = state.get("errors") or []
    if errors:
        lines += [
            "",
            "ERRORS ENCOUNTERED EARLIER IN THIS RUN:",
            *(f"- {e}" for e in errors),
        ]
    lines += ["", "YOUR TASK:", focus]
    return "\n".join(lines)


def subtask_stamp(state: dict, agent: AgentType, out: dict) -> dict:
    """State update marking this agent's subtasks done (with a result snippet)."""
    plan = state.get("plan")
    if not plan:
        return {}
    plan = json.loads(json.dumps(plan))  # copy: LangGraph channels are replaced
    for task in plan.get("subtasks", []):
        if task.get("agent") == agent.value and task.get("status") != "done":
            task["status"] = "done"
            task["result"] = (out.get("content") or "")[:500]
    return {"plan": plan}


def previous_context(state: dict, key: str, label: str) -> str:
    """Digest another agent's output for downstream agents."""
    value = state.get(key)
    if not value:
        return f"(no {label} yet)"
    content = value.get("content", "") if isinstance(value, dict) else str(value)
    return f"{label}:\n{content[:2500]}"
