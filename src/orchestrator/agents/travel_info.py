"""Travel Information Agent: weather, visa/entry, local rules, events."""
from __future__ import annotations

from orchestrator.agents import base
from orchestrator.schemas import AgentType, TraceEvent

SYSTEM = """You are the Travel Information Agent on a multi-agent travel-planning team.

Your job: destination-specific practical information.
Use tools:
- get_weather for the likely travel window
- get_visa_info for entry requirements (pass the traveler's nationality
  from the profile or request; if unknown, assume the reader's context
  and say so)
- search_wikipedia for local rules, safety notes, events, and etiquette

Then summarize:
1. Weather outlook and what to pack.
2. Visa/entry requirements and any verification caveats.
3. Key local rules, safety notes, etiquette relevant to the trip.
4. Notable events or seasonal conditions that affect planning.

IMPORTANT: If you cannot confidently verify visa/entry information, set
needs_attention=true with a clear attention_reason - do not guess.

When done, reply with ONLY a JSON object:
{"content": "<the summary as readable markdown>", "confidence": <0-1>,
 "sources": ["..."], "needs_attention": <bool>, "attention_reason": null}"""


def node(state: dict) -> dict:
    agent = AgentType.TRAVEL_INFO
    fallback = {
        "content": "Travel information research failed.",
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "travel_info agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(
            agent, SYSTEM, base.research_brief(state, "Research travel information for this trip.")
        ),
        agent,
        fallback,
    )
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={"confidence": out["confidence"], "needs_attention": out["needs_attention"]},
    ).model_dump()
    update: dict = {"travel_info": out}
    update.update(base.subtask_stamp(state, agent, out))
    if out["needs_attention"]:
        update["errors"] = [f"travel_info: {out.get('attention_reason') or 'needs attention'}"]
    update["trace"] = loop_events + [decision]
    return update
