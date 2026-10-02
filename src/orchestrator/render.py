"""Human-readable rendering of plans and itineraries (CLI + output files)."""
from __future__ import annotations

import json


def render_plan(plan: dict) -> str:
    req = plan.get("requirements", {})
    lines = [
        f"Goal: {plan.get('goal', '')}",
        (
            f"Trip: {req.get('days')} days, {req.get('travelers')} traveler(s), "
            f"destinations: {', '.join(req.get('destinations', [])) or 'n/a'}"
        ),
    ]
    if req.get("budget_total") is not None:
        lines.append(f"Budget: {req['budget_total']:,.0f} {req.get('currency', 'INR')}")
    prefs = []
    if req.get("food_preferences"):
        prefs.append("food: " + ", ".join(req["food_preferences"]))
    if req.get("pace"):
        prefs.append(f"pace: {req['pace']}")
    if req.get("interests"):
        prefs.append("interests: " + ", ".join(req["interests"]))
    if req.get("accommodation"):
        prefs.append("stays: " + req["accommodation"])
    if prefs:
        lines.append("Preferences: " + " | ".join(prefs))
    lines.append("Subtasks:")
    for task in plan.get("subtasks", []):
        deps = " (after " + ", ".join(task.get("depends_on", [])) + ")" if task.get("depends_on") else ""
        lines.append(f"  [{task.get('id')}] {task.get('agent')}: {task.get('description')}{deps}")
    lines.append(f"Supervisor confidence: {plan.get('confidence', 0):.2f}")
    return "\n".join(lines)


def render_itinerary(itinerary: dict, budget: dict | None = None) -> str:
    itinerary = itinerary or {}
    lines = [f"# {itinerary.get('title', 'Travel Itinerary')}", ""]
    if itinerary.get("summary"):
        lines += [itinerary["summary"], ""]

    if itinerary.get("bases"):
        lines.append("## Bases")
        for base in itinerary["bases"]:
            hotel = f" - {base['hotel']}" if base.get("hotel") else ""
            lines.append(
                f"- Day {base.get('check_in_day')}-{base.get('check_out_day')}: "
                f"{base.get('city')} ({base.get('nights')} nights){hotel}"
            )
        lines.append("")

    lines.append("## Day by day")
    for day in itinerary.get("days", []):
        city = f" - {day.get('city')}" if day.get("city") else ""
        theme = f" - {day['theme']}" if day.get("theme") else ""
        lines.append(f"### Day {day.get('day')}{city}{theme}")
        for act in day.get("activities", []):
            cost = f" (~{act['estimated_cost']:,.0f})" if act.get("estimated_cost") is not None else ""
            notes = f" - {act['notes']}" if act.get("notes") else ""
            lines.append(
                f"  - [{act.get('time_slot', '')}] {act.get('name')}{cost}{notes}"
            )
        if day.get("travel"):
            lines.append(f"  - travel: {day['travel']}")

    if budget and budget.get("items"):
        lines += ["", "## Budget"]
        for item in budget["items"]:
            lines.append(f"- {item.get('category')}: {item.get('amount'):,.0f} {item.get('currency', '')}")
            if item.get("details"):
                lines.append(f"    {item['details']}")
        limit = budget.get("budget_limit")
        status = "within budget" if budget.get("within_budget") else "OVER BUDGET"
        if limit is not None:
            lines.append(f"Total: {budget.get('total', 0):,.0f} (limit {limit:,.0f}) - {status}")
        else:
            lines.append(f"Total: {budget.get('total', 0):,.0f} - {status}")
        if budget.get("notes"):
            lines.append(f"Notes: {budget['notes']}")
    return "\n".join(lines)


def render_review_package(state: dict) -> str:
    """Everything a human needs to approve/modify/reject a trip plan."""
    plan = state.get("plan") or {}
    parts = [
        "=" * 60,
        "HUMAN REVIEW REQUIRED",
        "=" * 60,
        "",
        "REASON:" + " " + (state.get("escalation") or {}).get("reason", "escalation"),
        "",
        "ORIGINAL REQUEST:",
        state.get("request", ""),
        "",
        "PLAN:",
        render_plan(plan),
        "",
        "ITINERARY:",
        render_itinerary(state.get("itinerary"), state.get("budget_report")),
    ]
    review = state.get("review") or {}
    if review:
        parts += ["", "REVIEWER:", json.dumps(review, ensure_ascii=False, indent=2)]
    memories = state.get("memories") or []
    if memories:
        parts += ["", "MEMORIES USED:", *(f"- {m}" for m in memories)]
    parts.append("=" * 60)
    return "\n".join(parts)
