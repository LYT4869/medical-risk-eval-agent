from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from knowledge.retrieval.index import (CrossEncoderReranker, HybridRetriever,
                                       KnowledgeIndex,
                                       SentenceTransformerQueryEmbedding)


ROLE_SCOPES = {
    "patient": {"model_public", "clinical_patient"},
    "doctor": {"model_public", "model_technical", "clinical_patient",
               "clinical_professional"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a treeSem knowledge index")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).with_name("questions.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index = KnowledgeIndex(args.index)
    embedding = index.manifest["embedding"]
    reranker = index.manifest["reranker"]
    retriever = HybridRetriever(
        index,
        SentenceTransformerQueryEmbedding(embedding["model"], embedding["revision"]),
        CrossEncoderReranker(reranker["model"], reranker["revision"]),
        reranker_min_score=float(index.manifest["thresholds"]["reranker_min_score"]),
        all_scope_reranker_min_score=float(
            index.manifest["thresholds"]["all_scope_reranker_min_score"]),
        rrf_min_score=float(index.manifest["thresholds"]["rrf_min_score"]))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    negative_correct = 0
    negative_total = 0
    leakage = 0
    cases = []
    for question in questions:
        response = retriever.search(
            question["query"], question["role"], ROLE_SCOPES[question["role"]],
            question["scope"], 5)
        sources = [hit.chunk.source_id for hit in response.hits]
        expected = set(question["expected_source_ids"])
        if question["no_answer"]:
            negative_total += 1
            negative_correct += int(not sources)
        elif expected:
            recalls.append(len(expected.intersection(sources)) / len(expected))
            ranks = [sources.index(item) + 1 for item in expected if item in sources]
            reciprocal_ranks.append(0.0 if not ranks else 1.0 / min(ranks))
        if question["role"] == "patient":
            leakage += sum(hit.chunk.knowledge_scope in
                           {"model_technical", "clinical_professional"}
                           for hit in response.hits)
        cases.append({"id": question["id"], "sources": sources,
                      "retrieval_mode": response.retrieval_mode})
    report = {
        "index_version": index.version,
        "question_count": len(questions),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "negative_no_answer": negative_correct / negative_total,
        "cross_audience_leakage": leakage,
        "thresholds": {"recall_at_5": 0.85, "mrr_at_10": 0.75,
                       "negative_no_answer": 0.90,
                       "cross_audience_leakage": 0},
        "cases": cases,
    }
    output = args.output or args.index / "evaluation-baseline.json"
    output.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output.resolve().parent == args.index.resolve():
        manifest_path = args.index / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = output.read_bytes()
        manifest["files"][output.name] = {
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest_path.write_text(json.dumps(
            manifest, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")) + "\n", encoding="utf-8")
    index.close()
    if (report["recall_at_5"] < 0.85 or report["mrr_at_10"] < 0.75 or
            report["negative_no_answer"] < 0.90 or leakage != 0):
        raise SystemExit("knowledge evaluation thresholds were not met")


if __name__ == "__main__":
    main()
