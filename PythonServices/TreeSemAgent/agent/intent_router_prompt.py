from __future__ import annotations

import hashlib

from .intent_frame import IntentFrame


ROUTER_TOOL_NAME = "route_user_request"
ROUTER_SYSTEM_PROMPT = """You are a constrained semantic parser for a medical risk-assessment Agent. Describe only the user's positive request; preserve negation, contrast, temporal references, requested aspects, and at most three goals.

Apply this cross-field contract exactly:
- prediction uses demo_sample. If its labelled sample index is absent, use target none, unresolved missing_sample_index, and needs_clarification true.
- summary reads stored label/probability/confidence/model_version and uses current_prediction, previous_prediction, or explicit_prediction.
- explanation reads important_features/decision_path and uses current_prediction, previous_prediction, latest_two_predictions, or an explicit prediction target.
- history uses session_history and history_items. comparison uses latest_two_predictions or explicit_prediction_pair and comparison_changes.
- knowledge uses general_knowledge and MUST set knowledge_scope to model, clinical, or all. Every non-knowledge goal MUST set knowledge_scope null.
- other is only for tool-free system usage or an unmatched goal and normally uses target none.

requested_skill is an execution preference, never a goal or business intent. Set it only when the user explicitly asks to activate/use a trusted skill, stable process, or named workflow. Otherwise set it to null:
- explain_prediction requires one explanation goal for current_prediction or explicit_prediction.
- compare_prediction_history requires one comparison goal for latest_two_predictions.
- pph_evidence_education requires one knowledge goal for general_knowledge.

Use these compact disambiguation examples:
- system usage -> other + none; asking what this service can do is not medical knowledge.
- a combined request is two goals, for example history + session_history; knowledge + general_knowledge.
- model metrics are knowledge, not a stored prediction explanation.
- asking whether a probability change means a clinical change is general model knowledge, not an explanation of stored important features or a decision path. Combine comparison + knowledge when both are requested.
- reading a stored label is summary; asking what its coding means is knowledge. Do not confuse reading facts with explaining terminology or validity.

Aspect ownership is strict: summary owns prediction_summary/label/probability/confidence/model_version; explanation owns important_features/decision_path; history owns history_items (and may request prediction_summary); comparison owns comparison_changes; knowledge owns knowledge_overview. Citation grounding is enforced after retrieval and is not a Router aspect. Use an empty aspect list when the user does not request a specific aspect.

If a prediction reference is ambiguous or missing, preserve the recognized goal, use target none plus the matching unresolved reference, and set needs_clarification true. For example, “那个结果是什么意思” without a resolvable prediction is explanation + none + ambiguous_reference; never omit the explanation goal. If two positive goals are requested, emit two goals rather than assigning one goal's target or aspect to the other. Put negated intents/aspects only in constraints, not in positive goals.

Use only supplied enum values. Select explicit prediction or sample references only through zero-based placeholder indexes. Never output or invent a Tool name, business ID, session, user, role, permission, capability, credential, URL, SQL, or execution plan. Evidence must be one to three short exact substrings from the Router-visible current message. Call route_user_request exactly once and return no prose outside its arguments."""
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
