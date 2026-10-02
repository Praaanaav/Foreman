"""LLM access: model factory, usage capture, and JSON helpers.

All providers go through litellm so model names in config can point at
Groq, OpenAI, Anthropic, or any litellm-supported backend.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from orchestrator import config

log = logging.getLogger("orchestrator.llm")

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def get_model(role: str = "worker", temperature: float = 0.2) -> ChatLiteLLM:
    """Return a chat model for the given role.

    role="supervisor" uses SUPERVISOR_MODEL, everything else WORKER_MODEL.
    """
    model = config.SUPERVISOR_MODEL if role == "supervisor" else config.WORKER_MODEL
    return ChatLiteLLM(model=model, temperature=temperature)


def usage_of(msg: AIMessage) -> dict:
    """Token usage from an AIMessage, in a stable shape for tracing."""
    if getattr(msg, "usage_metadata", None):
        m = msg.usage_metadata
        return {
            "tokens_in": m.get("input_tokens", 0),
            "tokens_out": m.get("output_tokens", 0),
            "total_tokens": m.get("total_tokens", 0),
        }
    tu = (msg.response_metadata or {}).get("token_usage") or {}
    return {
        "tokens_in": tu.get("prompt_tokens", 0),
        "tokens_out": tu.get("completion_tokens", 0),
        "total_tokens": tu.get("total_tokens", 0),
    }


def extract_json(text: str):
    """Best-effort extraction of the first JSON object from LLM output.

    Handles ```json fences and prose around the JSON block.
    """
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in model output")


def parse_json(text: str, schema: type[T], model=None) -> T:
    """Parse LLM text into a pydantic model, with one self-repair attempt.

    If parsing fails and a chat model is provided, ask it to fix its own
    JSON before giving up.
    """
    try:
        return schema(**extract_json(text))
    except Exception as exc:
        if model is None:
            raise
        repair_prompt = (
            "Your previous JSON failed to parse: "
            f"{type(exc).__name__}: {exc}\n"
            f"Original output:\n{text[:3000]}\n"
            "Return ONLY the corrected JSON object. No prose, no code fences."
        )
        out = model.invoke([HumanMessage(content=repair_prompt)])
        return schema(**extract_json(out.content))
