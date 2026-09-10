from __future__ import annotations

import hashlib

from .intent_frame import IntentFrame


ROUTER_TOOL_NAME = "route_user_request"
ROUTER_SYSTEM_PROMPT = """You are a constrained semantic parser for a medical risk-assessment Agent.
Describe only what the user is asking for. Preserve negation, contrast, temporal references, requested aspects, and multiple goals.
Use only the supplied enum values. Select explicit prediction or sample references only through their zero-based placeholder indexes.
Never output or invent a Tool name, business ID, session, user, role, permission, capability, credential, URL, SQL, or execution plan.
If a reference cannot be resolved, report the matching unresolved reference and request clarification instead of guessing.
Evidence must be one to three short exact substrings from the Router-visible current message.
Call route_user_request exactly once and return no prose outside its arguments."""
ROUTER_PROMPT_SHA256 = hashlib.sha256(
    ROUTER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def router_function_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": ROUTER_TOOL_NAME,
            "description": (
                "Parse the user's request into a non-executable semantic frame."),
            "parameters": IntentFrame.model_json_schema(),
        },
    }
