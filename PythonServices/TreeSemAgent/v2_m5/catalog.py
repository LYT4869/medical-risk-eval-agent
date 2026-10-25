"""Small synthetic assertion catalog, NOT a general clinical knowledge base.

Each claim has two evidence-preserving textual variants; no LLM is allowed to
invent new clinical assertions or candidate facts in the P001 CI pathway.
"""
from __future__ import annotations

from collections import OrderedDict

FACTS = {
    "F-01": {
        "full": "患者档案包含一条患者自述类病史资料",
        "short": "存在患者自述病史资料",
        "evidence_ids": ["E-001"], "topic": "prior_history",
    },
    "F-02": {
        "full": "原始分娩记录中的估计失血量为250 mL",
        "short": "原始记录估计失血250 mL",
        "evidence_ids": ["E-002"], "topic": "source_review",
    },
    "F-03": {
        "full": "另一份转诊摘要记载约800 mL失血量",
        "short": "转诊摘要约800 mL",
        "evidence_ids": ["E-003"], "topic": "source_review",
    },
    "F-04": {
        "full": "上述两份来源记录的估计值不同，原因尚未核实",
        "short": "两份来源的差异尚未核实",
        "evidence_ids": ["E-002", "E-003"], "topic": "source_discrepancy",
    },
    "F-05": {
        "full": "当前专业模型未验证适用于产前任务，故本报告不提供风险分数",
        "short": "产前模型未获验证，本报告无风险分数",
        "evidence_ids": [], "topic": "model_applicability",
    },
    "F-06": {
        "full": "同一LIS检验记录经官方更正后的血红蛋白数值为130 g/L",
        "short": "LIS更正后Hb为130 g/L",
        "evidence_ids": ["E-005"], "topic": "source_correction",
    },
    "F-07": {
        "full": "旧版110 g/L的血红蛋白记录已被同源新版本取代，不能作为当前数值",
        "short": "旧版110 g/L已失效，不能作为当前值",
        "evidence_ids": ["E-004", "E-005"], "topic": "source_correction",
    },
    "F-08": {
        "full": "已有资料不能核实既往产后出血的正式诊断，不据此确认疾病",
        "short": "已有资料不足以确认既往产后出血诊断",
        "evidence_ids": ["E-001", "E-002", "E-003"], "topic": "prior_history",
    },
    "F-09": {
        "full": "现有调查保留来源差异及待复核项，这只是未签发的受限辅助草稿",
        "short": "来源差异待复核，当前仅为受限草稿",
        "evidence_ids": ["E-002", "E-003"], "topic": "limitations",
    },
    "F-10": {
        "full": "本合成调查没有检索经审核的指南或外部文献，不声称完成研究依据核对",
        "short": "本次未核对医学指南及外部文献",
        "evidence_ids": [], "topic": "knowledge_gap",
    },
}

# One paragraph per section in this P001 vertical slice. Real Writer is not
# restricted to these paragraph counts, but would require a real reviewer.
SECTION_PLAN = (
    ("patient_summary", "P-01", ("F-01", "F-06")),
    ("investigation", "P-02", ("F-02", "F-03", "F-04")),
    ("model_reference", "P-03", ("F-05",)),
    ("research_basis", "P-04", ("F-10",)),
    ("uncertainties", "P-05", ("F-07", "F-08")),
    ("overall_summary", "P-06", ("F-09",)),
)


def candidate_refs(fact_ids):
    return list(dict.fromkeys(
        ref for fid in fact_ids for ref in FACTS[fid]["evidence_ids"]
    ))


def paragraph_text(fact_ids, variant: str):
    return "；".join(FACTS[fid][variant] for fid in fact_ids) + "。"


def plan():
    return [
        {"section_id": section, "paragraph_id": pid, "fact_ids": list(facts)}
        for section, pid, facts in SECTION_PLAN
    ]
