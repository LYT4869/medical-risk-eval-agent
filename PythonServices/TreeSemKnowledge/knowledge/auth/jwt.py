from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from ..models import ALLOWED_SCOPES


@dataclass(frozen=True)
class KnowledgeClaims:
    actor_id: str
    actor_role: str
    session_id: str
    subject_user_id: str
    run_id: str
    scopes: frozenset[str]


def _decode(value: str, maximum: int) -> bytes:
    if len(value) > maximum * 2:
        raise ValueError("JWT segment too long")
    padding = "=" * ((4 - len(value) % 4) % 4)
    data = base64.urlsafe_b64decode(value + padding)
    if len(data) > maximum:
        raise ValueError("JWT segment too long")
    return data


def verify_knowledge_token(token: str, secret: str,
                           now: int | None = None) -> KnowledgeClaims:
    if len(secret) < 32 or len(token) > 8192:
        raise ValueError("invalid knowledge credential configuration")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid knowledge token")
    signing_input = f"{parts[0]}.{parts[1]}".encode()
    expected = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()).rstrip(b"=").decode()
    if not hmac.compare_digest(expected, parts[2]):
        raise ValueError("invalid knowledge token signature")
    header = json.loads(_decode(parts[0], 1024))
    claims: dict[str, Any] = json.loads(_decode(parts[1], 8192))
    current = int(time.time()) if now is None else now
    required = {"iss", "aud", "sub", "role", "session_id", "subject_user_id",
                "run_id", "scopes", "tools", "iat", "exp", "jti"}
    if set(claims) != required or header != {"alg": "HS256", "typ": "JWT"}:
        raise ValueError("invalid knowledge token shape")
    if claims["iss"] != "treesem-backend" or claims["aud"] != "treesem-knowledge":
        raise ValueError("invalid knowledge token audience")
    if not isinstance(claims["exp"], int) or claims["exp"] <= current:
        raise ValueError("knowledge token expired")
    if not isinstance(claims["iat"], int) or claims["iat"] > current + 60:
        raise ValueError("invalid knowledge token issued time")
    if claims["exp"] - claims["iat"] <= 0 or claims["exp"] - claims["iat"] > 300:
        raise ValueError("invalid knowledge token lifetime")
    if claims["role"] not in {"patient", "doctor"}:
        raise ValueError("invalid knowledge actor role")
    if claims["tools"] != ["search_medical_knowledge"]:
        raise ValueError("knowledge tool is out of scope")
    scopes = claims["scopes"]
    if not isinstance(scopes, list) or not scopes or not set(scopes).issubset(ALLOWED_SCOPES):
        raise ValueError("invalid knowledge scopes")
    if claims["role"] == "patient" and not set(scopes).issubset(
            {"model_public", "clinical_patient"}):
        raise ValueError("patient knowledge scope escalation")
    for name in ("sub", "session_id", "subject_user_id", "run_id", "jti"):
        if not isinstance(claims[name], str) or not claims[name] or len(claims[name]) > 128:
            raise ValueError("invalid knowledge identity claim")
    return KnowledgeClaims(claims["sub"], claims["role"], claims["session_id"],
                           claims["subject_user_id"], claims["run_id"],
                           frozenset(scopes))
