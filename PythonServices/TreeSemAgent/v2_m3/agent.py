"""One optional LLM/FC-compatible decision boundary, Fake trace by default.

The doctor-confirmed Contract, versioned Skill and risk profile shape bounded
context. Tools are actually executed only by the trusted Runtime/Guard.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from v2_m1.run_store import WorkflowConflict
from .policy import allowed_sources


class FakeInvestigationAgent:
    """Deterministic single-Agent policy used for durable P001 regression."""
    def decide(self, context: dict) -> dict:
        issue = context["issue"]
        if not context["timeline_loaded"]:
            return {"action": "tool", "tool": "get_patient_timeline"}
        required = allowed_sources(issue)
        observed = set(issue["evidence_ids"])
        for ref in sorted(required - observed):
            return {"action": "tool", "tool": "read_patient_record", "record_id": ref}
        return {"action": "close", "evidence_ids": sorted(required),
                "followup": "source_discrepancy" if issue["issue_id"] == "I-02"
                else None}


class OpenAICompatibleInvestigationAgent:
    """Optional **locally deployed** OpenAI-compatible chat-completion model.

    No default external endpoint, hidden API key, or fallback cloud service.
    A successful mock/stub test is not a validated real Qwen inference result.
    """
    SYSTEM = (
        "你是围产期合成 P001 调查工具决策器。只输出 JSON 对象，无 markdown。"
        "只能选择 action=tool (tool=get_patient_timeline/read_patient_record,"
        "可选 record_id) 或 action=close (evidence_ids=[...],followup=null 或"
        " source_discrepancy)。不得编写临床事实、诊断、患者身份或报告正文。"
        "根据当前 Issue 和已见 Evidence ID 选择下一步。证据冲突必须保留，"
        "I-02 close 时 followup=source_discrepancy；不得调用预测模型。"
    )

    def __init__(self, endpoint: str, model: str, bearer_token: str | None = None,
                 timeout: float = 15):
        parts = urlsplit(endpoint)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username or parts.password or parts.fragment
                or parts.query or (parts.scheme == "http" and
                    parts.hostname not in ("localhost", "127.0.0.1", "::1"))):
            raise ValueError("LLM endpoint must use loopback HTTP or HTTPS")
        if not model or not model.strip() or timeout <= 0 or timeout > 45:
            raise ValueError("invalid local LLM model or timeout")
        self.endpoint = endpoint
        self.model = model
        self.token = bearer_token
        self.timeout = timeout

    def decide(self, context: dict) -> dict:
        safe = {
            "contract_goal": str(context["contract_goal"])[:360],
            "skill": context["skill"],
            "risk_profile": context["risk_profile"],
            "issue": context["issue"],
            "evidence_summaries": context["evidence_summaries"],
            "timeline_loaded": context["timeline_loaded"],
            "available_tools": context["available_tools"],
            "remaining_tool_calls": context["remaining_tool_calls"],
        }
        req = Request(
            self.endpoint,
            data=json.dumps({
                "model": self.model, "temperature": 0,
                "messages": [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
                ],
            }, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": "Bearer " + self.token} if self.token else {}),
            },
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read(65536))
        raw = result["choices"][0]["message"]["content"]
        if not isinstance(raw, str) or len(raw) > 4096:
            raise WorkflowConflict("invalid LLM decision size")
        decision = json.loads(raw)
        if not isinstance(decision, dict):
            raise WorkflowConflict("LLM returned non-object decision")
        return decision
