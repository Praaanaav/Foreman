"""Run tracing: save, summarize, and render a trip-planning execution.

Every run produces a trace file under outputs/traces/<trip_id>.json with:
- the event timeline (agent decisions, LLM calls, tool calls, retries,
  escalations, human interventions)
- the full tool-call log from the ToolRegistry
- aggregate metrics (tokens, tool calls, latency, failures)

The trace explorer UI (phase 4) will render this file; for now the CLI
prints a compact tree via render_tree().
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator import config


def save_trace(trip_id: str, state: dict, tool_log: list) -> Path:
    trace_dir = config.OUTPUT_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trip_id": trip_id,
        "traveler_id": state.get("traveler_id"),
        "request": state.get("request"),
        "final_status": state.get("final_status"),
        "metrics": summarize(state.get("trace", []), tool_log),
        "events": state.get("trace", []),
        "tool_calls": [r.model_dump() for r in tool_log],
        "state_keys": sorted(state.keys()),
    }
    path = trace_dir / f"{trip_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def summarize(events: list[dict], tool_log: list) -> dict:
    llm_calls = [e for e in events if e.get("event") == "llm_call"]
    tool_events = [e for e in events if e.get("event") == "tool_call"]
    return {
        "llm_calls": len(llm_calls),
        "tokens_in": sum(e.get("detail", {}).get("tokens_in", 0) for e in llm_calls),
        "tokens_out": sum(e.get("detail", {}).get("tokens_out", 0) for e in llm_calls),
        "tool_calls": len(tool_log),
        "tool_failures": sum(1 for r in tool_log if not r.success),
        "retries": sum(1 for e in events if e.get("event") == "retry"),
        "escalations": sum(1 for e in events if e.get("event") == "escalation"),
        "human_interventions": sum(1 for e in events if e.get("event") == "human"),
        "agents": sorted({e.get("agent") for e in events if e.get("agent")}),
        "total_latency_ms": round(sum(e.get("latency_ms") or 0 for e in events), 1),
    }


def render_tree(events: list[dict]) -> str:
    """Compact execution tree: agent > llm_call/tool_call/decision lines."""
    lines: list[str] = []
    current_agent: str | None = None
    for e in events:
        agent = e.get("agent", "?")
        if agent != current_agent:
            lines.append(f"{agent}")
            current_agent = agent
        ev = e.get("event")
        if ev == "llm_call":
            d = e.get("detail", {})
            lines.append(
                f"  LLM {d.get('model', '?')} {e.get('latency_ms', 0):.0f}ms "
                f"(in {d.get('tokens_in', 0)} / out {d.get('tokens_out', 0)})"
            )
        elif ev == "tool_call":
            d = e.get("detail", {})
            status = "ok" if d.get("success") else f"FAIL: {d.get('error')}"
            lines.append(f"  tool {d.get('tool')} {status} {e.get('latency_ms', 0):.0f}ms")
        elif ev == "decision":
            detail = json.dumps(e.get("detail", {}), ensure_ascii=False)
            lines.append(f"  decision: {detail[:200]}")
        elif ev == "retry":
            lines.append(f"  RETRY {e.get('detail', {}).get('error', '')[:120]}")
        elif ev in ("escalation", "human", "memory"):
            lines.append(f"  {ev}: {json.dumps(e.get('detail', {}), ensure_ascii=False)[:200]}")
    return "\n".join(lines)
