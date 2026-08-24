from __future__ import annotations

import re


class PolicyViolation(RuntimeError):
    pass


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
            raise PolicyViolation("empty final answer")
        if not set(cited).issubset(available):
            raise PolicyViolation("response cited an unavailable prediction")
        mentioned = set(self._prediction_id.findall(answer))
        if not mentioned.issubset(available):
            raise PolicyViolation("response mentioned an unavailable prediction")
        resolved_predictions = list(dict.fromkeys(cited + sorted(mentioned)))
        if require_prediction_grounding and not resolved_predictions:
            resolved_predictions = sorted(available)
        if self._numeric_fact.search(answer) and not resolved_predictions:
            raise PolicyViolation("prediction facts require grounding identifiers")
        source_ids = set(cited_sources or [])
        source_available = available_sources or set()
        if not source_ids.issubset(source_available):
            raise PolicyViolation("response cited unavailable knowledge")
        mentioned_sources = set(self._citation_id.findall(answer))
        if not mentioned_sources.issubset(source_available):
            raise PolicyViolation("response mentioned unavailable knowledge")
        resolved_sources = source_ids | mentioned_sources
        if source_available and not resolved_sources:
            raise PolicyViolation("knowledge answer requires an explicit citation")
        return resolved_predictions, sorted(resolved_sources)
