from __future__ import annotations

import re


class PolicyViolation(RuntimeError):
    pass


class ResponsePolicy:
    _prediction_id = re.compile(r"pred_[0-9a-f]{32}")
    _numeric_fact = re.compile(r"(?:\d+(?:\.\d+)?\s*%|probability|概率|置信度|标签|label)", re.I)

    def validate(self, answer: str, cited: list[str], available: set[str]) -> None:
        if not answer.strip():
            raise PolicyViolation("empty final answer")
        if not set(cited).issubset(available):
            raise PolicyViolation("response cited an unavailable prediction")
        mentioned = set(self._prediction_id.findall(answer))
        if not mentioned.issubset(available):
            raise PolicyViolation("response mentioned an unavailable prediction")
        if self._numeric_fact.search(answer) and not cited:
            raise PolicyViolation("prediction facts require grounding identifiers")
