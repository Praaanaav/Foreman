"""Reviewer Agent: validates the plan against requirements and consistency."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator import llm
from orchestrator.agents import base
from orchestrator.schemas import AgentType, ReviewResult, TraceEvent

SYSTEM = """You are the Travel Reviewer, a strict quality gate on a multi-agent
travel-planning team. You have no tools; you validate from the provided state.

Check the itinerary and budget against the requirements and research:
1. COMPLETENESS: every requested day covered; budget covers all categories.
2. CONSISTENCY: cities/days/hotel bases align; transfers match transport
   research; no day physically impossible (travel time + activities).
3. BUDGET: total <= limit; flagged overruns must be real overruns.
4. PREFERENCES: food preferences, pace, "minimal hotel changes", interests
   actually respected (count hotel changes! count activities per day vs pace).
5. GAPS: missing research, unverified visa info, unsupported claims,
   or any agent marked needs_attention.

Verdicts:
- "approve": solid plan; issues may be minor.
- "reject": fixable problems - the Itinerary Agent will redo it with your
  specific feedback. Be concrete: list exactly what to change.
- "escalate": the plan cannot be safely fixed automatically (budget cannot be
  met, visa/entry info unverified, research conflicts, repeated failure, or
  consequential action needed). Put why in escalation_reason.

Reply with ONLY a JSON object:
{"verdict": "approve|reject|escalate", "issues": ["..."],
 "feedback": "<concrete instructions if rejecting>",
 "confidence": <0-1>, "escalation_reason": "<string or null>"}"""


def node(state: dict) -> dict:
    agent = AgentType.REVIEWER
    plan = state.get("plan") or {}
    digest = "\n\n".join(
        [
            "REQUIREMENTS:\n" + json.dumps(plan.get("requirements"), ensure_ascii=False, indent=2),
            base.previous_context(state, "destination_research", "destination research"),
            base.previous_context(state, "transport_research", "transport research"),
            base.previous_context(state, "accommodation_research", "accommodation research"),
            base.previous_context(state, "travel_info", "travel info"),
            "BUDGET:\n" + json.dumps(state.get("budget_report") or {}, ensure_ascii=False, indent=2),
            "ITINERARY:\n" + json.dumps(state.get("itinerary") or {}, ensure_ascii=False, indent=2),
        ]
    )
    if any(
        (state.get(k) or {}).get("needs_attention")
        for k in (
            "destination_research",
            "transport_research",
            "accommodation_research",
            "travel_info",
        )
    ):
        digest += "\n\nNOTE: at least one research agent flagged needs_attention - inspect its attention_reason."

    model = llm.get_model("worker")
    fallback = ReviewResult(
        verdict="escalate",
        issues=["Reviewer could not run after retries."],
        feedback="",
        confidence=0.0,
        escalation_reason="reviewer agent failed after retries",
    )

    def attempt():
        response = model.invoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=digest)]
        )
        return llm.parse_json(response.content, ReviewResult, model=model)

    review, retry_events = base.with_retries(attempt, agent, fallback.model_dump())

    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={
            "verdict": review["verdict"],
            "issues": review["issues"],
            "confidence": review["confidence"],
        },
    ).model_dump()
    update: dict = {"review": review}
    update["trace"] = retry_events + [decision]
    return update
