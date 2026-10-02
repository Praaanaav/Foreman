"""Budget Agent: totals everything and checks the budget limit."""
from __future__ import annotations

import json

from orchestrator.agents import base
from orchestrator.schemas import AgentType, BudgetReport, TraceEvent

SYSTEM = """You are the Budget Agent on a multi-agent travel-planning team.

Your job: produce a complete budget for the trip from the research so far.
Use convert_currency when research quotes are in a different currency than
the trip budget.

Build a BudgetReport JSON:
{
  "items": [{"category": "<flights|trains|local_transport|hotels|food|activities|visa|misc>",
              "amount": <number>, "currency": "<code>", "details": "<what it covers>"}, ...],
  "total": <sum in the budget currency>,
  "currency": "<budget currency>",
  "budget_limit": <limit or null>,
  "within_budget": <bool>,
  "notes": "<assumptions, and if over budget: where to trim, most to least impactful>"
}

Rules:
- Use the figures from the transportation and accommodation research for
  those categories; estimate food/activities/local transport sensibly for
  the trip length, pace, and group size.
- All amounts in ONE currency (the budget currency).
- within_budget = total <= budget_limit (true when there is no limit).
- If over budget, say exactly what to cut.

In your final reply, ONLY a JSON object:
{"content": "<the BudgetReport JSON as a string>", "confidence": <0-1>,
 "sources": ["..."], "needs_attention": <bool>, "attention_reason": null}"""


def node(state: dict) -> dict:
    agent = AgentType.BUDGET
    plan = state.get("plan") or {}
    brief = base.research_brief(
        state,
        "Calculate the complete trip budget from the research below.",
    )
    brief += (
        "\n\nTRANSPORT RESEARCH:\n"
        + base.previous_context(state, "transport_research", "transport")
        + "\n\nACCOMMODATION RESEARCH:\n"
        + base.previous_context(state, "accommodation_research", "accommodation")
        + "\n\nDESTINATION RESEARCH:\n"
        + base.previous_context(state, "destination_research", "destination")
    )

    fallback_report = {
        "items": [],
        "total": 0,
        "currency": plan.get("requirements", {}).get("currency", "INR"),
        "budget_limit": None,
        "within_budget": True,
        "notes": "Budget agent failed; treat figures as unavailable.",
    }
    fallback = {
        "content": json.dumps(fallback_report),
        "confidence": 0.0,
        "needs_attention": True,
        "attention_reason": "budget agent failed after retries",
        "sources": [],
    }
    out, loop_events = base.with_retries(
        lambda: base.run_tool_loop(agent, SYSTEM, brief),
        agent,
        fallback,
    )

    report = _parse_report(out)
    decision = TraceEvent(
        agent=agent.value,
        event="decision",
        detail={
            "total": report["total"],
            "budget_limit": report["budget_limit"],
            "within_budget": report["within_budget"],
        },
    ).model_dump()
    update: dict = {"budget_report": report}
    update.update(base.subtask_stamp(state, agent, out))
    if not report["within_budget"]:
        update["errors"] = [f"budget: total exceeds limit ({report['notes']})"]
    update["trace"] = loop_events + [decision]
    return update


def _parse_report(out: dict) -> dict:
    try:
        return BudgetReport(**json.loads(out["content"])).model_dump()
    except Exception:
        return {
            "items": [],
            "total": 0,
            "currency": "INR",
            "budget_limit": None,
            "within_budget": False,
            "notes": "Budget report could not be parsed: " + out.get("content", "")[:300],
        }
