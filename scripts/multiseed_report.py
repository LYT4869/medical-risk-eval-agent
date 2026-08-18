#!/usr/bin/env python3
"""Summarize explicitly selected, comparable treeSem multi-seed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


METRICS = ("auc", "auprc", "positive_f1", "balanced_accuracy", "accuracy",
           "sensitivity", "specificity", "brier_score", "ece_10_bin")
SEEDS = (19, 21, 42, 60, 99)


def signature(artifact: dict[str, Any]) -> str:
    args = dict(artifact.get("args") or {})
    ignored = {"seed", "device", "output_dir", "run_name", "raw_file"}
    payload = {key: str(value) for key, value in args.items() if key not in ignored}
    payload["model_config"] = artifact.get("model_config")
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260818)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean()
                        for _ in range(10_000)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(artifact: dict[str, Any], frame: pd.DataFrame) -> dict[str, float]:
    args = dict(artifact.get("args") or {})
    labels = frame.iloc[:, -1].replace({-1: 0})
    train_x, test_x, _, test_y = train_test_split(
        frame.iloc[:, 2:-4], labels,
        test_size=float(args.get("test_size", 0.2)),
        random_state=int(args["seed"]),
        stratify=labels if bool(args.get("stratify_split", False)) else None)
    values = StandardScaler().fit(train_x).transform(test_x).astype(np.float32)
    state = artifact["model_state"]
    with torch.inference_mode():
        tensor = torch.from_numpy(values)
        hidden = torch.relu(torch.nn.functional.linear(
            tensor, state["fc1.weight"].cpu(), state["fc1.bias"].cpu()))
        mu = torch.nn.functional.linear(
            hidden, state["fc21.weight"].cpu(), state["fc21.bias"].cpu())
        classifier = torch.relu(torch.nn.functional.linear(
            mu, state["main.0.weight"].cpu(), state["main.0.bias"].cpu()))
        probability = torch.softmax(torch.nn.functional.linear(
            classifier, state["main.3.weight"].cpu(),
            state["main.3.bias"].cpu()), dim=1)[:, 1].numpy()
    expected = np.asarray(test_y, dtype=np.int64)
    predicted = (probability >= 0.5).astype(np.int64)
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.reshape(-1))
    calibration = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9 else
            probability < edges[index + 1])
        if selected.any():
            calibration += float(selected.mean()) * abs(
                float(expected[selected].mean()) - float(probability[selected].mean()))
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "auc": float(roc_auc_score(expected, probability)),
        "auprc": float(average_precision_score(expected, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "brier_score": float(np.mean(np.square(probability - expected))),
        "ece_10_bin": calibration,
        "positive_f1": float(f1_score(expected, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs=5, type=Path)
    parser.add_argument("--raw-file", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("build/reports/m11-five-seed.json"))
    args = parser.parse_args()
    rows = []
    signatures = set()
    feature_sets: list[set[int]] = []
    frame = pd.read_csv(args.raw_file)
    for path in args.artifacts:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(dict(artifact.get("args") or {}).get("seed", -1))
        metrics = artifact.get("metrics")
        if seed not in SEEDS or not isinstance(metrics, dict):
            raise SystemExit(f"invalid multi-seed artifact: {path.name}")
        signatures.add(signature(artifact))
        selected = set(int(value) for value in np.asarray(
            artifact.get("t_star", artifact.get("u_star"))).reshape(-1))
        feature_sets.append(selected)
        recomputed = evaluate(artifact, frame)
        rows.append({"seed": seed, "artifact": path.name,
                     "metrics": recomputed,
                     "stored_metric_alignment": {
                         name: abs(float(metrics[name]) - recomputed[target]) <= 1e-6
                         for name, target in {
                             "accuracy": "accuracy", "auc": "auc",
                             "balanced_accuracy": "balanced_accuracy",
                             "positive_f1": "positive_f1",
                             "positive_recall": "sensitivity"}.items()},
                     "tree_depth": int(artifact["tree"].tree_.max_depth),
                     "selected_feature_count": len(selected)})
    if sorted(row["seed"] for row in rows) != list(SEEDS):
        raise SystemExit("artifacts must contain seeds 19, 21, 42, 60 and 99 exactly once")
    if len(signatures) != 1:
        raise SystemExit("multi-seed artifacts do not share one experiment configuration")
    pairwise_jaccard = []
    for left in range(len(feature_sets)):
        for right in range(left + 1, len(feature_sets)):
            union = feature_sets[left] | feature_sets[right]
            pairwise_jaccard.append(len(feature_sets[left] & feature_sets[right]) /
                                    len(union) if union else 1.0)
    summary = {}
    for name in METRICS:
        values = np.asarray([row["metrics"][name] for row in rows])
        summary[name] = {"mean": float(values.mean()),
                         "sample_std": float(values.std(ddof=1)),
                         "bootstrap_95_ci": bootstrap_ci(values)}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "configuration_sha256": next(iter(signatures)),
        "runs": sorted(rows, key=lambda row: row["seed"]), "summary": summary,
        "explanation_stability": {
            "measure": "pairwise selected-feature Jaccard",
            "mean": float(np.mean(pairwise_jaccard)),
            "minimum": float(np.min(pairwise_jaccard)),
        },
        "selection_policy": "report_only; never select a model using test metrics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": list(SEEDS)}, sort_keys=True))


if __name__ == "__main__":
    main()
