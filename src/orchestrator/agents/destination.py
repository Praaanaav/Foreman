"""Destination Research Agent: destinations, attractions, what to see."""
from __future__ import annotations

from orchestrator.agents import base
from orchestrator.schemas import AgentType, TraceEvent

SYSTEM = """You are the Destination Research Agent on a multi-agent travel-planning team.

Your job: research the destination cities/regions for this trip.
Use tools to ground your research in real facts:
- search_wikipedia for background, season, and culture
- search_attractions to find the major attractions and landmarks
- get_weather / calc_travel_time where helpful

Then summarize:
1. The city breakdown that fits the trip (which cities, how many days each,
   given the pace and "minimal hotel changes" style if stated).
2. For each city: 5-8 must-see attractions with a one-line why.
3. Best time-of-year notes and any travel season warnings.
4. Local practicalities (transport between the cities, rough distances/times).

Be specific and factual. When done, reply with ONLY a JSON object:
{"content": "<the summary above as readable markdown>", "confidence": <0-1>,
 "sources": ["<tool or page names>"], "needs_attention": <bool>,
 "attention_reason": null}
Set needs_attention=true only if you could not verify something important."""


def node(state: dict) -> dict:
    agent = AgentType.DESTINATION
    fallback = {
        "content": "Destination research failed.",
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "destination agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(
            agent, SYSTEM, base.research_brief(state, "Research the destinations for this trip.")
        ),
        agent,
        fallback,
    )
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={"confidence": out["confidence"], "needs_attention": out["needs_attention"]},
    ).model_dump()
    update: dict = {"destination_research": out}
    update.update(base.subtask_stamp(state, agent, out))
    if out["needs_attention"]:
        update["errors"] = [f"destination: {out.get('attention_reason') or 'needs attention'}"]
    update["trace"] = loop_events + [decision]
    return update
