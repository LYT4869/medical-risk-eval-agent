from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .routing_types import RequestScope


def _normalize(message: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", message).casefold().split())


def _has(message: str, markers: tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)


_EDUCATIONAL_CONTEXT = (
    "如何防止", "如何检测", "如何识别", "为什么不应该", "为什么不能",
    "一般情况下", "一般性", "通常如何",
    "有哪些风险", "有什么风险", "原则是什么", "科普",
    "how to prevent", "how to detect", "what controls prevent",
    "why should", "general guidance", "generally",
)
_DEFENSIVE_INSTRUCTION = (
    "不要编造", "不得编造", "不能编造", "避免编造", "拒绝编造",
    "不要伪造", "不得伪造", "防止伪造", "do not fabricate",
    "don't fabricate", "must not fabricate", "do not invent",
    "don't invent", "must not invent",
)

_FABRICATION_ACTION = (
    "伪造", "编造", "编一个", "包装成真实", "假装", "fabricate",
    "随便生成", "凭空生成", "invent", "pretend",
)
_SECURITY_FACT = (
    "预测id", "预测 id", "预测", "概率", "引用", "真实引用", "虚假引用",
    "决策路径", "工具已经成功", "tool succeeded", "prediction id",
    "prediction", "probability", "citation", "decision path",
)
_MEDICAL_EVIDENCE = (
    "治疗依据", "专家建议", "医院方案", "医疗依据",
    "医院内部产后出血方案",
    "treatment evidence", "expert recommendation", "medical evidence",
)
_UNAVAILABLE_EVIDENCE = (
    "没有查到", "没有 evidence", "没有evidence", "没有也要",
    "未检索到", "不存在", "未收录", "缺失", "absent", "unindexed",
    "unavailable", "does not contain", "not contain",
)

_DISCLOSURE_ACTION = (
    "回显", "输出", "显示", "打印", "泄露", "告诉我", "给我",
    "print", "show", "reveal", "echo", "disclose", "return",
)
_SECRET_TARGET = (
    "authorization头", "authorization 头", "api key", "apikey",
    "access token", "refresh token", "密码", "密钥", "系统提示词",
    "system prompt", "configured secret", "all configured secrets",
)
_PRIVILEGE_ACTION = (
    "绑定", "构造", "生成", "访问", "读取", "查询", "无视", "忽略",
    "绕过", "越过", "假装", "不要鉴权", "扩大", "提升", "关", "关闭", "禁用",
    "调用", "bypass", "disable",
    "turn off", "ignore", "override", "bind", "access", "read",
)
_PRIVILEGE_TARGET = (
    "其他会话", "别的会话", "所有患者", "其他患者", "另一名患者",
    "未分配患者", "未授权患者", "别人的结果", "管理员接口", "管理员工具",
    "工具白名单", "能力令牌", "capability token", "safety gate",
    "guardrail", "other session", "all patients", "other patient",
    "admin api", "administrator api", "tool allowlist", "tool whitelist",
)
_ACCESS_CONTROL_TARGET = (
    "访问控制", "受保护的预测", "access control", "protected prediction",
)
_BYPASS_ACTION = ("绕过", "规避", "bypass", "circumvent")
_EXECUTE_INJECTION = (
    "照它执行", "按它执行", "遵照检索内容", "follow it", "execute it",
)
_VERIFICATION_ESCALATION = (
    "unverified feedback 标成 doctor verified",
    "unverified feedback改成doctor verified",
    "mark unverified feedback as doctor verified",
)

_PERSONAL_CONTEXT = (
    "为我", "给我", "替我", "我的情况", "我应该", "我是否", "我现在",
    "让我", "我今天", "我对药物", "根据我的", "for me", "for my situation",
    "for my case", "give me", "should i", "whether i", "my medication",
    "i am", "i have",
)
_INDIVIDUALIZED_TARGET = (
    "个体化", "个人化", "individualized", "personalized",
)
_DIRECTIVE_ACTION = (
    "直接选", "直接决定", "替我决定", "告诉我该", "告诉我是否",
    "制定", "修改", "开具", "处方", "具体", "精确", "exact",
    "specific", "prescribe", "decide", "choose", "modify", "change",
)
_MEDICATION_ACTION = (
    "药和剂量", "药物剂量", "具体剂量", "精确剂量", "用药方案",
    "停药", "换药", "加药", "减药", "停止用药", "处方",
    "drug and dose", "exact dose", "exact dosage", "medication",
    "medication regimen",
    "stop my medication", "change my medication", "prescription",
)
_PROCEDURE_ACTION = (
    "是否手术", "要不要手术", "做哪种手术", "采用哪种治疗",
    "应该马上手术", "手术决定", "whether i need surgery", "which surgery",
    "which procedure", "choose a treatment",
)
_DIAGNOSIS_ACTION = (
    "确定诊断", "直接诊断", "下诊断", "diagnose", "diagnosis",
    "definitive diagnosis",
)
_TREATMENT_PLAN = (
    "治疗计划", "治疗方案", "用药计划", "treatment plan",
    "treatment regimen",
)
_OUTCOME_GUARANTEE = (
    "保证有效", "保证治好", "一定有效", "肯定有效", "绝对有效",
    "guarantee", "guaranteed to work", "certain to work",
)
_UNSUPPORTED_SOURCE_DEMAND = (
    "也要", "仍然提供", "照样提供", "provide a source even if",
    "provide evidence even if",
)
_SOURCE_OR_EVIDENCE = (
    "来源", "资料", "证据", "引用", "source", "evidence", "citation",
)

_BLEEDING = (
    "大量出血", "持续大出血", "产后出血", "heavy bleeding",
    "bleeding heavily", "active bleeding", "postpartum hemorrhage",
    "postpartum haemorrhage",
)
_INSTABILITY = (
    "头晕", "晕倒", "意识不清", "昏厥", "昏倒", "快要倒下",
    "feel faint", "fainting", "faint", "altered consciousness",
    "unconscious", "passing out",
)
_CURRENT_STATE = (
    "现在", "当前", "立刻", "正在", "还头晕", "我快要", "我正在",
    "快要", "患者当前", "right now", "currently", "today", "i am", "i have",
    "the patient has", "active",
)
_AVOID_CARE_OR_SELF_TREAT = (
    "不想联系急救", "不联系急救", "不去医院", "自己处理", "自行处理",
    "居家处理", "home treatment", "manage it alone", "without emergency",
    "avoid emergency", "do not call emergency",
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    refusal_scope: RequestScope | None = None
    reason: str | None = None


class SafetyPolicy:
    """Classify finite high-risk concepts without granting permissions."""

    def evaluate(self, message: str) -> SafetyDecision:
        normalized = _normalize(message)

        emergency = self._emergency_reason(normalized)
        if emergency is not None:
            return SafetyDecision(
                False, RequestScope.MEDICAL_REFUSAL, emergency)

        medical = self._medical_reason(normalized)
        if medical is not None:
            return SafetyDecision(
                False, RequestScope.MEDICAL_REFUSAL, medical)

        if self._is_explicitly_defensive(normalized):
            return SafetyDecision(True)

        security = self._security_reason(normalized)
        if security is not None:
            return SafetyDecision(
                False, RequestScope.SECURITY_ABUSE, security)

        return SafetyDecision(True)

    @staticmethod
    def _is_explicitly_defensive(message: str) -> bool:
        return _has(message, _DEFENSIVE_INSTRUCTION) or _has(
            message, _EDUCATIONAL_CONTEXT)

    @staticmethod
    def _emergency_reason(message: str) -> str | None:
        unstable = _has(message, _INSTABILITY)
        current = _has(message, _CURRENT_STATE)
        self_treatment = _has(message, _AVOID_CARE_OR_SELF_TREAT)
        if (_has(message, _BLEEDING) and unstable and
                (current or self_treatment)):
            return "urgent_medical_symptoms"
        if unstable and current and self_treatment:
            return "urgent_medical_symptoms"
        return None

    @staticmethod
    def _medical_reason(message: str) -> str | None:
        fabricated = _has(message, _FABRICATION_ACTION)
        unavailable = _has(message, _UNAVAILABLE_EVIDENCE)
        if fabricated and _has(message, _MEDICAL_EVIDENCE):
            return "fabricated_medical_evidence"
        if unavailable and (_has(message, (
                "quote", "protocol", "治疗依据", "专家建议", "医院内部",
                "医院方案")) or (
                    _has(message, _UNSUPPORTED_SOURCE_DEMAND) and
                    _has(message, _SOURCE_OR_EVIDENCE))):
            return "fabricated_medical_evidence"

        personal = _has(message, _PERSONAL_CONTEXT)
        directive = _has(message, _DIRECTIVE_ACTION)
        direct_medication_change = _has(message, (
            "停药", "换药", "加药", "减药", "停止用药",
            "stop my medication", "change my medication"))
        if ((personal or _has(message, _INDIVIDUALIZED_TARGET)) and
                _has(message, _MEDICATION_ACTION) and
                (directive or direct_medication_change)):
            return "personalized_medication_decision"
        if personal and _has(message, _PROCEDURE_ACTION):
            return "personalized_procedure_decision"
        if personal and _has(message, _DIAGNOSIS_ACTION):
            return "diagnostic_certainty"
        if personal and directive and _has(message, _TREATMENT_PLAN):
            return "personalized_treatment_decision"
        if personal and _has(message, _OUTCOME_GUARANTEE):
            return "treatment_guarantee"
        return None

    @staticmethod
    def _security_reason(message: str) -> str | None:
        if (_has(message, _DISCLOSURE_ACTION) and
                _has(message, _SECRET_TARGET)):
            return "credential_exfiltration"
        if _has(message, _VERIFICATION_ESCALATION):
            return "verification_forgery"
        if (_has(message, _BYPASS_ACTION) and
                _has(message, _ACCESS_CONTROL_TARGET)):
            return "privilege_escalation"
        if (_has(message, _PRIVILEGE_ACTION) and
                _has(message, _PRIVILEGE_TARGET)):
            return "privilege_escalation"
        if (_has(message, _EXECUTE_INJECTION) and
                _has(message, ("safety gate", "guardrail", "工具白名单"))):
            return "instruction_override"
        if (_has(message, _FABRICATION_ACTION) and
                _has(message, _SECURITY_FACT)):
            return "fact_fabrication"
        return None
