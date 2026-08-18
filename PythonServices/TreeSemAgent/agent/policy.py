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
                 available_sources: set[str] | None = None) -> None:
        if not answer.strip():
            raise PolicyViolation("empty final answer")
        if not set(cited).issubset(available):
            raise PolicyViolation("response cited an unavailable prediction")
        mentioned = set(self._prediction_id.findall(answer))
        if not mentioned.issubset(available):
            raise PolicyViolation("response mentioned an unavailable prediction")
        if self._numeric_fact.search(answer) and not cited:
            raise PolicyViolation("prediction facts require grounding identifiers")
        source_ids = set(cited_sources or [])
        source_available = available_sources or set()
        if not source_ids.issubset(source_available):
            raise PolicyViolation("response cited unavailable knowledge")
        mentioned_sources = set(self._citation_id.findall(answer))
        if not mentioned_sources.issubset(source_available):
            raise PolicyViolation("response mentioned unavailable knowledge")
        if source_ids != mentioned_sources:
            raise PolicyViolation("knowledge citations must be explicit in the answer")
