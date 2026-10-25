"""One Writer call producing a paragraph draft; source IDs are candidates only.

FakeWriter is the regression fixture. OpenAICompatibleWriter is a *potential*
local model adapter; CI does not deploy a real model or claim semantic safety.
"""
from __future__ import annotations

import json

from v2_m1.run_store import WorkflowConflict
from v2_m3.agent import OpenAICompatibleInvestigationAgent
from .catalog import SECTION_PLAN, candidate_refs, paragraph_text


class FakeParagraphWriter:
    def write(self, context: dict) -> list[dict]:
        return [
            {"section_id": section, "paragraph_id": pid,
             "text": paragraph_text(facts, "full"),
             "candidate_evidence_refs": candidate_refs(facts),
             "fact_ids": list(facts)}
            for section, pid, facts in SECTION_PLAN
        ]


class LocalParagraphWriter(OpenAICompatibleInvestigationAgent):
    """Optional local structured Writer; same loopback/HTTPS endpoint policy.

    Validation is deliberately *outside* this model call; unsafe output is
    rejected, never repaired by guessing a patient's missing facts.
    """
    SYSTEM = (
        "你只编写P001合成报告的6个段落，不生成新的医学事实。"
        "按提供的已审核事实和段落清单输出JSON数组。每条包含"
        "section_id,paragraph_id,fact_ids,text,candidate_evidence_refs。"
        "可改写行文但不添加诊断、风险评分或临床处置建议。"
        "来源引用仅是候选，后续独立核验。"
    )

    def write(self, context: dict) -> list[dict]:
        if "available_claims" not in context or "section_plan" not in context:
            raise WorkflowConflict("Writer must receive bounded verified context")
        # We reuse the vetted network transport rules but do not call decide():
        # that method expects a Tool action and would misparse paragraph arrays.
        from urllib.request import Request, urlopen
        req = Request(self.endpoint,
            data=json.dumps({"model": self.model, "temperature": 0,
                 "messages": [
                     {"role": "system", "content": self.SYSTEM},
                     {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                 ]}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **(
                {"Authorization": "Bearer " + self.token} if self.token else {})},
            method="POST")
        with urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read(65536))
        raw = payload["choices"][0]["message"]["content"]
        if not isinstance(raw, str) or len(raw) > 30000:
            raise WorkflowConflict("invalid Writer output size")
        result = json.loads(raw)
        if not isinstance(result, list) or len(result) != 6:
            raise WorkflowConflict("Writer returned unexpected section count")
        return result
