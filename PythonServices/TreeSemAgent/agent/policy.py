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

    def validate(self, answer: str, cited: list[str], available: set[str],
                 cited_sources: list[str] | None = None,
                 available_sources: set[str] | None = None,
                 require_prediction_grounding: bool = False
                 ) -> tuple[list[str], list[str]]:
        if not answer.strip():
            raise PolicyViolation("empty_answer", "empty final answer")
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
        if self._numeric_fact.search(answer) and not resolved_predictions:
            raise PolicyViolation(
                "missing_prediction_grounding",
                "prediction facts require grounding identifiers")
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
        return resolved_predictions, sorted(resolved_sources)
