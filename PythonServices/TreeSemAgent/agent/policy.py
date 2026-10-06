from __future__ import annotations

import re


class PolicyViolation(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


SAFE_POLICY_FALLBACK = (
    "我无法提供未经可信工具结果验证的预测、概率或引用。"
    "请重新描述问题，或先获取可验证的预测和知识资料。"
)

SAFE_SECURITY_REFUSAL = (
    "我不能帮助伪造预测或引用、绕过权限，"
    "或访问其他患者的记录。"
)

SAFE_MEDICAL_REFUSAL = (
    "我不能根据有限信息提供个体化处方、具体药物剂量或确定诊断，"
    "也不能引用知识库中不存在的医疗资料。请咨询专业医生；"
    "如有紧急症状，请立即联系当地急救服务。"
)

SAFE_EMERGENCY_RESPONSE = (
    "这可能是紧急情况，请立即联系当地急救服务或尽快前往急诊，"
    "不要等待本系统检索资料或给出诊断。treeSem 不能替代现场医疗处置。"
)


class ResponsePolicy:
    _prediction_id = re.compile(r"pred_[0-9a-f]{32}")
    _citation_id = re.compile(r"cite_[0-9a-f]{20}")
    _numeric_fact = re.compile(r"(?:\d+(?:\.\d+)?\s*%|probability|概率|置信度|标签|label)", re.I)
    # A narrow tripwire for concrete patient assertions, not a medical-content
    # classifier. General terminology and model metrics are different evidence.
    _patient_reference = re.compile(
        r"(?:你的|您的|你(?:这次|本次|上次|之前)|"
        r"(?:当前|本次|这次)(?=预测|结果)|该患者|该病人|\byour\b|\bthis patient\b)",
        re.I)
    _prediction_term = re.compile(
        r"概率|置信度|标签|风险|预测|结果|positive_probability|"
        r"\b(?:probability|confidence|label|risk|prediction)\b", re.I)
    _numeric_value = re.compile(
        r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?:\s*%)?(?![A-Za-z0-9_])")
    _class_value_statement = re.compile(
        r"(?:标签|预测|结果|\blabel\b|\bprediction\b)\s*"
        r"(?:为|是|=|:|：|\bis\b)\s*"
        r"(?:阳性|阴性|高风险|低风险|\bpositive\b|\bnegative\b)", re.I)
    _visible_reference = re.compile(r"(?<![A-Za-z0-9_])(?:pred|cite)_[A-Za-z0-9_-]+")
    _protocol_line = re.compile(r'''(?<![A-Za-z0-9_])["']?grounding_(?:prediction|source)_ids["']?\s*:\s*\[''')

    def _has_patient_value(self, answer: str) -> bool:
        # A comma can continue the same patient assertion. Keep that scope, but
        # require the value to follow the patient subject and prediction term:
        # an earlier aggregate metric followed by "not your probability" is
        # not a patient measurement. Decimal points are not sentence boundaries.
        for sentence in re.split(r"[。!?;；\n]|(?<!\d)\.(?!\d)", answer):
            subject = self._patient_reference.search(sentence)
            if subject is None:
                continue
            tail = sentence[subject.end():]
            term = self._prediction_term.search(tail)
            if term and (self._numeric_value.search(tail[term.end():]) or
                         self._class_value_statement.search(tail)):
                return True
        return False

    def validate(self, answer: str, cited: list[str], available: set[str],
                 cited_sources: list[str] | None = None,
                 available_sources: set[str] | None = None,
                 require_prediction_grounding: bool = False,
                 knowledge_only: bool = False
                 ) -> tuple[list[str], list[str]]:
        if not answer.strip():
            raise PolicyViolation("empty_answer", "empty final answer")
        if self._protocol_line.search(answer):
            raise PolicyViolation("invalid_final_response", "answer contains protocol fields")
        for ident in self._visible_reference.findall(answer):
            syntax = self._prediction_id if ident.startswith("pred_") else self._citation_id
            if syntax.fullmatch(ident) is None:
                # Do not guess which complete ID a truncated reference meant.
                raise PolicyViolation("invalid_body_reference", "answer contains an invalid reference")
        if not set(cited).issubset(available):
            raise PolicyViolation(
                "unavailable_prediction", "response cited an unavailable prediction")
        mentioned = set(self._prediction_id.findall(answer))
        if not mentioned.issubset(available):
            raise PolicyViolation(
                "unavailable_prediction", "response mentioned an unavailable prediction")
        resolved_predictions = list(dict.fromkeys(cited + sorted(mentioned)))
        if require_prediction_grounding and not resolved_predictions:
            resolved_predictions = sorted(available)
        source_ids = set(cited_sources or [])
        source_available = available_sources or set()
        if not source_ids.issubset(source_available):
            raise PolicyViolation(
                "unavailable_knowledge", "response cited unavailable knowledge")
        mentioned_sources = set(self._citation_id.findall(answer))
        if not mentioned_sources.issubset(source_available):
            raise PolicyViolation(
                "unavailable_knowledge", "response mentioned unavailable knowledge")
        resolved_sources = source_ids | mentioned_sources
        if source_available and not resolved_sources:
            raise PolicyViolation(
                "missing_knowledge_citation",
                "knowledge answer requires an explicit citation")
        # Only the caller's validated task can grant this scope, never the LLM
        # envelope or the mere presence of a retrieved citation. All referenced
        # IDs have already been checked against this run's actual Tool evidence.
        grounded_knowledge = knowledge_only and bool(resolved_sources)
        if not resolved_predictions and (
                self._has_patient_value(answer) or
                (self._numeric_fact.search(answer) and not grounded_knowledge)):
            raise PolicyViolation(
                "missing_prediction_grounding",
                "prediction facts require grounding identifiers")
        return resolved_predictions, sorted(resolved_sources)
