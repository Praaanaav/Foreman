"""Accommodation Agent: hotels/stays by budget, location, preferences."""
from __future__ import annotations

from orchestrator.agents import base
from orchestrator.schemas import AgentType, TraceEvent

SYSTEM = """You are the Accommodation Agent on a multi-agent travel-planning team.

Your job: recommend stays for each base city.
Use tools:
- estimate_hotels for options per city (synthetic estimates: realistic
  planning figures, clearly label them as estimates)
- search_wikipedia if you need area/district context

Respect the requirements: total budget, traveler count, stated
accommodation style, and any "minimal hotel changes" preference (prefer
fewer, longer stays in one base city over many short ones).

Then summarize per base city:
1. 2-3 options across budget bands with estimated total cost for the stay.
2. A recommended pick with reasons (location vs budget vs comfort).
3. Check-in/check-out day suggestions consistent with fewer moves.

When done, reply with ONLY a JSON object:
{"content": "<the summary as readable markdown>", "confidence": <0-1>,
 "sources": ["..."], "needs_attention": <bool>, "attention_reason": null}"""


def node(state: dict) -> dict:
    agent = AgentType.ACCOMMODATION
    fallback = {
        "content": "Accommodation research failed.",
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "accommodation agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(
            agent, SYSTEM, base.research_brief(state, "Research accommodation for this trip.")
        ),
        agent,
        fallback,
    )
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={"confidence": out["confidence"], "needs_attention": out["needs_attention"]},
    ).model_dump()
    update: dict = {"accommodation_research": out}
    update.update(base.subtask_stamp(state, agent, out))
    if out["needs_attention"]:
        update["errors"] = [f"accommodation: {out.get('attention_reason') or 'needs attention'}"]
    update["trace"] = loop_events + [decision]
    return update
