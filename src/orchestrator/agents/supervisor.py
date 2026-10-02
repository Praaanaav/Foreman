"""Travel Supervisor: parses the request, retrieves context, decomposes the plan."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator import llm
from orchestrator.agents import base
from orchestrator.schemas import (
    AgentType,
    TravelPlan,
    TraceEvent,
    SubTask,
    TripRequirements,
)

SYSTEM = """You are the Travel Supervisor on a multi-agent travel-planning team.
Specialists: destination, transportation, accommodation, travel_info, budget, itinerary.

Given the user request, the traveler profile, and retrieved memories, produce a
task-decomposition JSON object with exactly this shape:
{
  "goal": "<one sentence>",
  "requirements": {
    "origin": "<city or null>",
    "destinations": ["<country/region>", ...],
    "days": <int>,
    "travelers": <int>,
    "budget_total": <number or null>,
    "currency": "<ISO code, default INR>",
    "food_preferences": ["..."],
    "pace": "slow" | "moderate" | "fast",
    "interests": ["..."],
    "accommodation": "<preference or null>",
    "constraints": ["..."]
  },
  "subtasks": [
    {"id": "t1", "description": "...", "agent": "destination", "depends_on": [],
     "required_inputs": ["requirements"], "expected_output": "...", "complexity": "medium"},
    ...
  ],
  "confidence": <0.0-1.0>
}

Rules:
- ALWAYS include one subtask for each of: destination, transportation,
  accommodation, travel_info, budget, itinerary.
- The four research agents (destination, transportation, accommodation,
  travel_info) must have empty depends_on so they can run in parallel.
- budget depends on the four research subtasks; itinerary depends on budget.
- Where the request is ambiguous, make sensible defaults and note them in
  requirements.constraints.
Return ONLY the JSON object - no prose, no code fences."""


def default_plan(request: str) -> TravelPlan:
    """Heuristic fallback if the supervisor LLM is unavailable."""
    m_days = re.search(r"(\d+)\s*-?\s*day", request, re.IGNORECASE)
    m_pax = re.search(r"\bfor\s+(\d+)\s+(?:people|persons|travellers|travelers)\b", request, re.IGNORECASE)
    m_budget = re.search(r"\d[\d,]*", request)
    dest = re.search(r"(?:trip to|visit|to)\s+([A-Z][a-zA-Z]+)", request)
    requirements = TripRequirements(
        origin=None,
        destinations=[dest.group(1)] if dest else ["unspecified"],
        days=int(m_days.group(1)) if m_days else 7,
        travelers=int(m_pax.group(1)) if m_pax else 2,
        budget_total=float(m_budget.group(0).replace(",", "")) if m_budget else None,
    )
    subtasks = [
        SubTask(id=f"t{i}", description=desc, agent=agent)
        for i, (desc, agent) in enumerate(
            [
                ("Research destinations and attractions", AgentType.DESTINATION),
                ("Research transportation options", AgentType.TRANSPORTATION),
                ("Research accommodation options", AgentType.ACCOMMODATION),
                ("Research weather/visa/local rules", AgentType.TRAVEL_INFO),
                ("Calculate the trip budget", AgentType.BUDGET),
                ("Build the day-by-day itinerary", AgentType.ITINERARY),
            ],
            start=1,
        )
    ]
    return TravelPlan(
        goal="Fallback heuristic plan (supervisor LLM unavailable).",
        requirements=requirements,
        subtasks=subtasks,
        confidence=0.3,
    )


def _brief(state: dict) -> str:
    parts = [
        "USER REQUEST:",
        state.get("request", ""),
        "",
    ]
    profile = state.get("profile") or {}
    if profile:
        parts += ["TRAVELER PROFILE:", json.dumps(profile, ensure_ascii=False), ""]
    memories = state.get("memories") or []
    if memories:
        parts += ["RETRIEVED TRAVELER MEMORIES:", *(f"- {m}" for m in memories), ""]
    parts += [
        "PREVIOUS TRIPS:",
        state.get("previous_trips") or "(none)",
        "",
        "Produce the task-decomposition JSON now.",
    ]
    return "\n".join(parts)


def node(state: dict) -> dict:
    from orchestrator import config

    model = llm.get_model("supervisor")
    fallback = default_plan(state.get("request", ""))

    def attempt():
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=_brief(state)),
            ]
        )
        return llm.parse_json(response.content, TravelPlan, model=model)

    plan, retry_events = base.with_retries(attempt, AgentType.SUPERVISOR, fallback)

    event = TraceEvent(
        agent=AgentType.SUPERVISOR.value,
        event="decision",
        detail={
            "subtasks": [s.id for s in plan.subtasks],
            "requirements": plan.requirements.model_dump(),
            "confidence": plan.confidence,
        },
    ).model_dump()

    return {
        "plan": plan.model_dump(),
        "attempts": state.get("attempts", 0),
        "trace": retry_events + [event],
    }
