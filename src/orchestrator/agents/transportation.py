"""Transportation Agent: flights, trains, transfers, local transport."""
from __future__ import annotations

from orchestrator.agents import base
from orchestrator.schemas import AgentType, TraceEvent

SYSTEM = """You are the Transportation Agent on a multi-agent travel-planning team.

Your job: research how the travelers get around.
Use tools to ground your estimates:
- estimate_flights for inter-city/international legs (synthetic estimates:
  treat them as realistic planning figures, clearly label them as estimates)
- calc_travel_time for ground travel between cities
- search_wikipedia for the local transport situation (rail networks, etc.)

Then summarize:
1. Inbound / intercity / return legs with mode, rough duration and
   estimated cost in the trip currency (per person and total for the group).
2. Recommended ground transfer between each pair of cities (train/car/bus),
   duration and cost.
3. Local transport guidance per city (how to move around day to day, costs).
4. Practical tips: passes, booking lead times.

When done, reply with ONLY a JSON object:
{"content": "<the summary as readable markdown>", "confidence": <0-1>,
 "sources": ["..."], "needs_attention": <bool>, "attention_reason": null}"""


def node(state: dict) -> dict:
    agent = AgentType.TRANSPORTATION
    fallback = {
        "content": "Transportation research failed.",
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "transportation agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(
            agent, SYSTEM, base.research_brief(state, "Research transportation for this trip.")
        ),
        agent,
        fallback,
    )
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={"confidence": out["confidence"], "needs_attention": out["needs_attention"]},
    ).model_dump()
    update: dict = {"transport_research": out}
    update.update(base.subtask_stamp(state, agent, out))
    if out["needs_attention"]:
        update["errors"] = [f"transportation: {out.get('attention_reason') or 'needs attention'}"]
    update["trace"] = loop_events + [decision]
    return update
