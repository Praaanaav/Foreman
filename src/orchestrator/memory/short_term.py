"""Short-term trip memory.

During a planning session the LangGraph state *is* the shared short-term
memory: every agent reads and writes the same `TravelState` (requirements,
research results, budget, itinerary drafts, errors, reviewer feedback).

This module provides a stable view of that state. When we move to Redis
(phase 2), a `RedisTripMemory` implementing the same shape is what the
rest of the system would talk to instead of the in-memory state.
"""
from __future__ import annotations

# Keys that make up the short-term trip memory, grouped for inspection.
SHORT_TERM_GROUPS: dict[str, list[str]] = {
    "requirements": ["request", "plan"],
    "research": [
        "destination_research",
        "transport_research",
        "accommodation_research",
        "travel_info",
    ],
    "planning": ["budget_report", "itinerary"],
    "validation": ["review", "attempts", "errors"],
    "human_loop": ["escalation", "human_decision"],
}


def snapshot_short_term(state: dict) -> dict:
    """Grouped, present-only view of the trip's shared working memory."""
    snap: dict[str, dict] = {}
    for group, keys in SHORT_TERM_GROUPS.items():
        values = {k: state[k] for k in keys if k in state and state[k] is not None}
        if values:
            snap[group] = values
    return snap
