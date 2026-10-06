from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Protocol


class QueryEmbeddingProvider(Protocol):
    def encode_query(self, text: str) -> list[float]: ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("embeddings must contain finite values")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("embeddings must have positive norms")
    return math.fsum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm)


def find_similarity_flags(
        candidates: list[dict[str, object]],
        frozen: list[dict[str, object]],
        provider: QueryEmbeddingProvider, *,
        threshold: float) -> list[dict[str, object]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between zero and one")
    candidate_vectors = [
        provider.encode_query(str(item["message"])) for item in candidates]
    frozen_vectors = [
        provider.encode_query(str(item["message"])) for item in frozen]
    result = []
    for index, (candidate, vector) in enumerate(zip(
            candidates, candidate_vectors)):
        neighbors: list[tuple[float, str, str, str]] = []
        for other_index, (other, other_vector) in enumerate(zip(
                candidates, candidate_vectors)):
            if index == other_index:
                continue
            neighbors.append((
                cosine_similarity(vector, other_vector),
                str(other["candidate_id"]), "candidate",
                str(other["expected_scope"])))
        for other, other_vector in zip(frozen, frozen_vectors):
            neighbors.append((
                cosine_similarity(vector, other_vector),
                str(other["case_id"]), "frozen",
                str(other["expected_scope"])))
        if not neighbors:
            continue
        score, neighbor_id, neighbor_set, neighbor_scope = max(
            neighbors, key=lambda item: (item[0], item[2], item[1]))
        if score + 1e-12 < threshold:
            continue
        scope = str(candidate["expected_scope"])
        result.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_scope": scope,
            "neighbor_id": neighbor_id,
            "neighbor_scope": neighbor_scope,
            "neighbor_set": neighbor_set,
            "same_scope": scope == neighbor_scope,
            "cosine_similarity": round(score, 8),
        })
    return result


def _load_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
            payload.get("cases"), list):
        raise ValueError(f"invalid routing corpus: {path}")
    return payload["cases"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flag semantic near-duplicates in routing candidates")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from agent.onnx_embedding_provider import OnnxEmbeddingProvider
    from agent.routing_artifact import load_routing_artifact
    from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry

    manifest = json.loads(
        (args.artifact / "manifest.json").read_text(encoding="utf-8"))
    registry = TaskRegistry.load(args.tasks, SUPPORTED_DOMAIN_TOOLS)
    artifact = load_routing_artifact(
        args.artifact,
        task_registry_path=args.tasks,
        thresholds_path=args.thresholds,
        registry=registry,
        expected_backend="onnx_fp32",
        expected_model_id=manifest["source_model_id"],
        expected_revision=manifest["source_model_revision"],
    )
    candidates = _load_cases(args.candidates)
    frozen = _load_cases(args.frozen)
    flags = find_similarity_flags(
        candidates, frozen, OnnxEmbeddingProvider(artifact),
        threshold=args.threshold)
    report = {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "frozen_count": len(frozen),
        "review_threshold": args.threshold,
        "flag_count": len(flags),
        "cross_scope_flag_count": sum(
            not bool(item["same_scope"]) for item in flags),
        "flags": flags,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "flags"}, ensure_ascii=False, indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
