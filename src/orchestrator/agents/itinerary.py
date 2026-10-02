"""Itinerary Agent: builds the day-by-day plan from all research."""
from __future__ import annotations

import json

from orchestrator.agents import base
from orchestrator.schemas import AgentType, Itinerary, TraceEvent

SYSTEM = """You are the Itinerary Agent on a multi-agent travel-planning team.

Your job: combine all research into a realistic, budget-aware, day-by-day
itinerary that respects the traveler's preferences (pace, food, hotel-change
limits, interests).

Use calc_travel_time to sanity-check transfers between cities; do not pack
more into a day than the pace allows (slow pace = 1-2 major activities/day).
Include realistic travel time, and keep hotel changes minimal when requested.

Build an Itinerary JSON:
{
  "title": "<trip title>",
  "bases": [{"city": "...", "nights": n, "check_in_day": d, "check_out_day": d, "hotel": "..."}, ...],
  "days": [{"day": 1, "city": "...", "theme": "...",
             "activities": [{"name": "...", "time_slot": "morning|afternoon|evening|all-day",
                              "estimated_cost": <num|null>, "notes": "..."}],
             "travel": "<transfer to next base, or null>"}, ...],
  "summary": "<3-5 sentence overview incl. total estimated cost vs budget>"
}

Rules:
- Number of days must equal the requested trip length; days are contiguous 1..N.
- Every day must have at least one activity or an explicit travel/free day.
- Respect food preferences (e.g. vegetarian) in activity/food notes.

In your final reply, ONLY a JSON object:
{"content": "<the Itinerary JSON as a string>", "confidence": <0-1>,
 "sources": ["..."], "needs_attention": <bool>, "attention_reason": null}"""


def node(state: dict) -> dict:
    agent = AgentType.ITINERARY
    plan = state.get("plan") or {}
    requirements = plan.get("requirements") or {}

    brief = base.research_brief(
        state,
        "Build the day-by-day itinerary for this trip.",
    )
    brief += (
        "\n\nDESTINATION RESEARCH:\n"
        + base.previous_context(state, "destination_research", "destination")
        + "\n\nTRANSPORT RESEARCH:\n"
        + base.previous_context(state, "transport_research", "transport")
        + "\n\nACCOMMODATION RESEARCH:\n"
        + base.previous_context(state, "accommodation_research", "accommodation")
        + "\n\nTRAVEL INFO:\n"
        + base.previous_context(state, "travel_info", "travel info")
        + "\n\nBUDGET REPORT:\n"
        + json.dumps(state.get("budget_report") or {}, ensure_ascii=False)
    )

    # Replan / human-modify paths: carry the instruction into the brief.
    review = state.get("review") or {}
    if review.get("verdict") == "reject":
        brief += "\n\nREVIEWER FEEDBACK (fix these problems):\n" + review.get("feedback", "")
    human = state.get("human_decision") or {}
    if human.get("action") == "modify" and human.get("modifications"):
        brief += "\n\nHUMAN CHANGES REQUESTED:\n" + human["modifications"]

    fallback = {
        "content": "Itinerary generation failed.",
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "itinerary agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(agent, SYSTEM, brief),
        agent,
        fallback,
    )

    itinerary = _parse_itinerary(out)
    replanned = bool(review.get("verdict") == "reject" or human.get("action") == "modify")
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={
            "days": len(itinerary.get("days", [])),
            "requested_days": requirements.get("days"),
            "replan": replanned,
        },
    ).model_dump()
    update: dict = {"itinerary": itinerary}
    if replanned:
        update["attempts"] = state.get("attempts", 0) + 1
    update.update(base.subtask_stamp(state, agent, out))
    update["trace"] = loop_events + [decision]
    return update


def _parse_itinerary(out: dict) -> dict:
    try:
        return Itinerary(**json.loads(out["content"])).model_dump()
    except Exception:
        return {
            "title": "(unparsed itinerary)",
            "bases": [],
            "days": [],
            "summary": "Itinerary JSON could not be parsed: " + out.get("content", "")[:300],
        }
