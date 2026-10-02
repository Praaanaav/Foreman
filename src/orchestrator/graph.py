"""The LangGraph travel-planning state machine.

    START
      -> load_context      (traveler profile + long-term memory retrieval)
      -> supervisor        (parse requirements, decompose into subtasks)
      -> [destination, transportation, accommodation, travel_info]  (parallel)
      -> budget
      -> itinerary
      -> reviewer
         ├─ approve & within budget -> finalize
         ├─ reject (retries left)   -> itinerary (replan with feedback)
         └─ escalate / budget over  -> escalate
                interrupt() -> human decision
                ├─ approve   -> finalize
                ├─ take_over -> finalize (human-authored itinerary)
                ├─ modify    -> itinerary (with requested changes)
                └─ reject    -> supervisor (re-plan, retries capped)
      finalize -> END  (render, save outputs + trace, extract long-term memory)
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from orchestrator import config, llm, registry
from orchestrator.agents import (
    accommodation,
    budget,
    destination,
    itinerary,
    reviewer,
    supervisor,
    travel_info,
    transportation,
)
from orchestrator.agents.base import _now
from orchestrator.memory.long_term import LongTermMemory
from orchestrator.observability.trace import save_trace
from orchestrator.render import render_itinerary
from orchestrator.schemas import (
    TraceEvent,
    TravelState,
)

import orchestrator.travel_tools  # noqa: F401 - ensure tools are registered

log = logging.getLogger("orchestrator.graph")


def _event(agent: str, event: str, detail: dict) -> dict:
    return TraceEvent(agent=agent, event=event, detail=detail, timestamp=_now()).model_dump()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def load_context(state: TravelState) -> dict:
    memory = LongTermMemory()
    traveler = state.get("traveler_id", config.DEFAULT_TRAVELER)
    request = state.get("request", "")
    profile = memory.get_profile(traveler)
    memories = [m["text"] for m in memory.get_memories(traveler, request, k=5)]
    previous = memory.previous_trips_summary(traveler)
    return {
        "profile": profile,
        "memories": memories,
        "previous_trips": previous,
        "trace": [
            _event(
                "load_context",
                "memory",
                {
                    "profile_keys": list(profile.keys()),
                    "memories_retrieved": len(memories),
                    "previous_trips": previous.count("- "),
                },
            )
        ],
    }


def escalate_node(state: TravelState) -> dict:
    review = state.get("review") or {}
    budget_report = state.get("budget_report") or {}
    if budget_report and not budget_report.get("within_budget", True):
        reason = (
            f"Itinerary exceeds the user's budget "
            f"({budget_report.get('total'):,.0f} vs limit {budget_report.get('budget_limit'):,.0f})."
        )
    else:
        reason = review.get("escalation_reason") or "Reviewer flagged issues that need human judgment."

    payload = {
        "reason": reason,
        "request": state.get("request"),
        "requirements": (state.get("plan") or {}).get("requirements"),
        "itinerary": state.get("itinerary"),
        "budget": budget_report,
        "review_issues": review.get("issues", []),
        "reviewer_confidence": review.get("confidence"),
        "memories_used": state.get("memories", []),
        "errors_so_far": state.get("errors", []),
        "proposed_action": "Approve as-is, request modifications, reject for re-planning, or take over and write the itinerary yourself.",
    }
    decision = interrupt(payload)
    return {
        "escalation": payload,
        "human_decision": decision,
        "trace": [
            _event("escalate", "escalation", {"reason": reason}),
            _event("human", "human", {"decision": decision}),
        ],
    }


MEMORY_EXTRACT_SYSTEM = """You are the memory-extraction module of a travel-planning system.
From the completed trip below, extract durable traveler information.
Reply with ONLY a JSON object:
{"profile_update": {"budget_style": "...", "food": ["..."], "pace": "...",
   "accommodation": "...", "interests": ["..."], "traveler_count": <int or null>},
 "memories": [{"text": "<one useful lesson/preference/fact for future trips>",
               "tags": ["..."], "importance": <0.3-1.0>}]
Keep 2-6 memories. Only include things genuinely useful for future trips."""


def finalize_node(state: TravelState) -> dict:
    trip_id = state.get("trip_id", "unknown")
    traveler = state.get("traveler_id", config.DEFAULT_TRAVELER)
    human = state.get("human_decision") or {}
    itinerary_data = state.get("itinerary") or {}
    budget_report = state.get("budget_report")

    if human.get("action") == "take_over" and human.get("modifications"):
        final_md = "## Final Itinerary (human takeover)\n\n" + human["modifications"]
        status = "completed_human_takeover"
    elif itinerary_data.get("days"):
        final_md = render_itinerary(itinerary_data, budget_report)
        status = "completed"
    else:
        final_md = (
            "## Planning stopped\n\n"
            "The planning loops were exhausted (retries/escalations). "
            "See the trace file for the full execution record."
        )
        status = "aborted"

    # Save the itinerary file
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    itinerary_path = config.OUTPUT_DIR / f"itinerary_{trip_id}.md"
    itinerary_path.write_text(final_md, encoding="utf-8")

    # Long-term memory: save trip record + extract learnings
    memory_events: list[dict] = []
    memory = LongTermMemory()
    memory.save_trip(
        traveler,
        trip_id,
        {
            "title": itinerary_data.get("title") or state.get("request", "")[:80],
            "request": state.get("request"),
            "requirements": (state.get("plan") or {}).get("requirements"),
            "created_at": _now(),
            "summary": (itinerary_data.get("summary") or "")[:500],
            "status": status,
        },
    )
    memory_events.append(_event("memory", "memory", {"trip_saved": trip_id}))
    try:
        model = llm.get_model("worker", temperature=0.0)
        extraction_brief = "\n".join(
            [
                "REQUEST:",
                state.get("request", ""),
                "",
                "REQUIREMENTS:",
                json.dumps((state.get("plan") or {}).get("requirements"), ensure_ascii=False),
                "",
                "FINAL ITINERARY SUMMARY:",
                (itinerary_data.get("summary") or "")[:800],
                "",
                "REVIEW ISSUES (if any):",
                json.dumps((state.get("review") or {}).get("issues", []), ensure_ascii=False),
            ]
        )
        response = model.invoke(
            [
                SystemMessage(content=MEMORY_EXTRACT_SYSTEM),
                HumanMessage(content=extraction_brief),
            ]
        )
        extracted = json.loads(llm.extract_json(response.content))
        profile = {**(state.get("profile") or {}), **extracted.get("profile_update", {})}
        memory.save_profile(traveler, profile)
        for mem in extracted.get("memories", []):
            memory.add_memory(
                traveler,
                mem.get("text", "").strip(),
                tags=mem.get("tags", []),
                importance=float(mem.get("importance", 0.5)),
                source=trip_id,
            )
        memory_events.append(
            _event(
                "memory",
                "memory",
                {"memories_added": len(extracted.get("memories", []))},
            )
        )
    except Exception as exc:
        log.warning("memory extraction failed: %s", exc)
        memory_events.append(_event("memory", "retry", {"error": str(exc)}))

    # Trace
    state_with_status = {**state, "final_status": status}
    trace_path = save_trace(trip_id, state_with_status, registry.registry.call_log)

    return {
        "final_itinerary": final_md,
        "final_status": status,
        "trace": memory_events
        + [_event("finalize", "decision", {"status": status, "trace_file": str(trace_path)})],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_review(state: TravelState) -> str:
    review = state.get("review") or {}
    verdict = review.get("verdict")
    budget_report = state.get("budget_report") or {}

    # Hard rule: a plan that busts the user's budget must go to a human.
    if budget_report and budget_report.get("budget_limit") is not None and not budget_report.get(
        "within_budget", True
    ):
        return "escalate"
    if verdict == "escalate":
        return "escalate"
    if verdict == "reject":
        if state.get("attempts", 0) >= config.MAX_REPLANS:
            return "escalate"
        return "replan"
    if verdict != "approve":
        # Unknown verdict: fail safe, let a human look at it.
        return "escalate"
    return "finalize"


def route_after_human(state: TravelState) -> str:
    decision = state.get("human_decision") or {}
    action = decision.get("action")
    if action in ("approve", "take_over"):
        return "finalize"
    # modify / reject: re-plan if we still have budget, else stop gracefully
    if state.get("attempts", 0) >= config.MAX_REPLANS:
        return "finalize"
    return "itinerary" if action == "modify" else "supervisor"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

RESEARCH_NODES = {
    "destination": destination.node,
    "transportation": transportation.node,
    "accommodation": accommodation.node,
    "travel_info": travel_info.node,
}


def build_graph(checkpointer=None):
    graph = StateGraph(TravelState)
    graph.add_node("load_context", load_context)
    graph.add_node("supervisor", supervisor.node)
    for name, fn in RESEARCH_NODES.items():
        graph.add_node(name, fn)
    graph.add_node("budget", budget.node)
    graph.add_node("itinerary", itinerary.node)
    graph.add_node("reviewer", reviewer.node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "supervisor")
    graph.add_edge("supervisor", "destination")
    graph.add_edge("supervisor", "transportation")
    graph.add_edge("supervisor", "accommodation")
    graph.add_edge("supervisor", "travel_info")
    for name in RESEARCH_NODES:
        graph.add_edge(name, "budget")
    graph.add_edge("budget", "itinerary")
    graph.add_edge("itinerary", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"finalize": "finalize", "replan": "itinerary", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "escalate",
        route_after_human,
        {"finalize": "finalize", "itinerary": "itinerary", "supervisor": "supervisor"},
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


if __name__ == "__main__":
    g = build_graph()
    print(g.get_graph().draw_mermaid())
