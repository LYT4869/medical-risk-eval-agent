from __future__ import annotations

from .rule_evidence import (
    IntentAction,
    IntentObject,
    IntentReference,
    RuleEvidence,
)
from .routing_types import RequestScope, RoutingDecision, RoutingSource


_STORED_REFERENCES = {
    IntentReference.CURRENT_PREDICTION,
    IntentReference.PRIOR_PREDICTION,
}


class RuleIntentResolver:
    def resolve(self, evidence: RuleEvidence) -> RoutingDecision | None:
        if evidence.explicit_skill:
            return self._decision(RequestScope.SKILL)

        candidates = self._candidates(evidence)
        include_summary = (
            RequestScope.EXPLANATION in candidates and
            IntentObject.PREDICTION_FACT in evidence.objects)

        if RequestScope.COMPARISON in candidates:
            candidates.discard(RequestScope.HISTORY)
            candidates.discard(RequestScope.SUMMARY)
        if RequestScope.EXPLANATION in candidates:
            candidates.discard(RequestScope.SUMMARY)

        if len(candidates) > 1:
            return RoutingDecision(
                RequestScope.UNKNOWN,
                RoutingSource.RULE,
                reason="rule_compositional",
            )
        if candidates:
            return self._decision(
                next(iter(candidates)), include_summary=include_summary)
        if evidence.vague_reference:
            return RoutingDecision(
                RequestScope.UNKNOWN,
                RoutingSource.RULE,
                reason="rule_ambiguous_reference",
            )
        return None

    @staticmethod
    def _candidates(evidence: RuleEvidence) -> set[RequestScope]:
        actions = evidence.actions
        objects = evidence.objects
        references = evidence.references
        stored_reference = bool(references & _STORED_REFERENCES)

        candidates: set[RequestScope] = set()
        if ((IntentAction.PREDICT in actions and
             (IntentObject.DEMO_SAMPLE in objects or
              RequestScope.PREDICTION in evidence.registry_scopes)) or
                (RequestScope.PREDICTION in evidence.registry_scopes and
                 IntentObject.DEMO_SAMPLE in objects)):
            candidates.add(RequestScope.PREDICTION)
        if (IntentAction.READ in actions and
                IntentObject.PREDICTION_FACT in objects and
                stored_reference):
            candidates.add(RequestScope.SUMMARY)
        if (((IntentAction.EXPLAIN in actions and
              (IntentObject.PREDICTION_RECORD in objects or
               IntentObject.PREDICTION_FACT in objects)) or
             IntentObject.EXPLANATION_DETAIL in objects) and
                (stored_reference or
                 IntentReference.EXPLICIT_SAMPLE in references or
                 IntentReference.MULTIPLE_PREDICTIONS in references)):
            candidates.add(RequestScope.EXPLANATION)
        if (IntentAction.LIST in actions and
                IntentObject.HISTORY in objects):
            candidates.add(RequestScope.HISTORY)
        if (IntentAction.COMPARE in actions and
                (IntentReference.MULTIPLE_PREDICTIONS in references or
                 IntentReference.PRIOR_PREDICTION in references or
                 IntentObject.HISTORY in objects)):
            candidates.add(RequestScope.COMPARISON)
        knowledge_action = (
            IntentAction.RETRIEVE in actions or
            (IntentAction.EXPLAIN in actions and not stored_reference) or
            (IntentAction.READ in actions and not stored_reference))
        if knowledge_action and IntentObject.KNOWLEDGE in objects:
            candidates.add(RequestScope.KNOWLEDGE)
        return candidates

    @staticmethod
    def _decision(scope: RequestScope, *,
                  include_summary: bool = False) -> RoutingDecision:
        return RoutingDecision(
            scope=scope,
            source=RoutingSource.RULE,
            include_summary=include_summary,
        )
