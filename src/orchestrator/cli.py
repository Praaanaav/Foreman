"""End-to-end travel planning CLI.

Usage:
    python -m orchestrator.cli "Plan a 10-day Japan trip for two people with a 150000 INR budget. Vegetarian, slow travel, cultural experiences, minimal hotel changes."
    python -m orchestrator.cli --request-file request.txt --traveler alice
    python -m orchestrator.cli "..." --yes     # auto-approve escalations (CI/demo)

The run streams a compact execution log, pauses at human-review interrupts,
then prints the final itinerary and where outputs/trace were saved.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid

from langgraph.types import Command

from orchestrator import config, render
from orchestrator.graph import build_graph
from orchestrator.observability.trace import render_tree
from orchestrator.render import render_review_package
from orchestrator.schemas import HumanAction, HumanDecision


def _log(level: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {level:9s} {msg}", flush=True)


def _prompt_decision() -> HumanDecision:
    while True:
        choice = input(
            "\nDecision [a]pprove / [m]odify / [r]eject / [t]ake-over > "
        ).strip().lower()
        if choice in ("a", "approve"):
            return HumanDecision(action=HumanAction.APPROVE)
        if choice in ("m", "modify"):
            changes = input("Describe the changes you want (one line): ").strip()
            if not changes:
                print("No changes given; try again.")
                continue
            return HumanDecision(action=HumanAction.MODIFY, modifications=changes)
        if choice in ("r", "reject"):
            reason = input("Why reject? (fed back to the supervisor): ").strip()
            return HumanDecision(
                action=HumanAction.REJECT,
                modifications=reason or "Rejected for re-planning.",
            )
        if choice in ("t", "takeover", "take_over"):
            print("Paste the full itinerary to adopt (end with a line containing only '.'):")
            lines = []
            while True:
                line = input()
                if line.strip() == ".":
                    break
                lines.append(line)
            return HumanDecision(
                action=HumanAction.TAKE_OVER, modifications="\n".join(lines)
            )
        print("Please pick a, m, r or t.")


def run_planning(request: str, traveler: str, auto_yes: bool = False) -> dict:
    trip_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    graph = build_graph()
    cfg = {"configurable": {"thread_id": trip_id}}

    _log("start", f"trip {trip_id} | traveler={traveler}")
    _log("info", f"request: {request}")
    state = graph.invoke(
        {"trip_id": trip_id, "traveler_id": traveler, "request": request},
        cfg,
    )

    # Handle human-review interrupts (there may be more than one).
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print(render_review_package({**state, "escalation": payload}))
        _log("human", f"escalation: {payload.get('reason')}")
        if auto_yes:
            decision = HumanDecision(action=HumanAction.APPROVE)
            _log("human", "auto-approve (--yes)")
        else:
            decision = _prompt_decision()
        state = graph.invoke(Command(resume=decision.model_dump()), cfg)

    _log("done", f"status: {state.get('final_status')}")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Foreman travel planning CLI")
    parser.add_argument("request", nargs="?", help="The travel request")
    parser.add_argument("--request-file", help="Read the travel request from a file")
    parser.add_argument("--traveler", default=config.DEFAULT_TRAVELER)
    parser.add_argument("--yes", action="store_true", help="Auto-approve escalations")
    args = parser.parse_args(argv)

    request = args.request
    if args.request_file:
        request = open(args.request_file, encoding="utf-8").read().strip()
    if not request:
        parser.error("provide a request as an argument or via --request-file")

    logging.basicConfig(level=logging.WARNING)
    state = run_planning(request, args.traveler, auto_yes=args.yes)

    print("\n" + "=" * 60)
    print("PLAN (supervisor decomposition)")
    print("=" * 60)
    if state.get("plan"):
        print(render.render_plan(state["plan"]))

    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    print(render_tree(state.get("trace", [])))

    print("\n" + "=" * 60)
    print("FINAL ITINERARY")
    print("=" * 60)
    print(state.get("final_itinerary", "(none)"))

    trip_id = state.get("trip_id")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = config.OUTPUT_DIR / "traces" / f"{trip_id}.json"
    print(f"\nItinerary file: {config.OUTPUT_DIR / f'itinerary_{trip_id}.md'}")
    print(f"Trace file:     {trace_path}")
    return 0 if state.get("final_status") in ("completed", "completed_human_takeover") else 1


if __name__ == "__main__":
    sys.exit(main())
