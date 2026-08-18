#!/usr/bin/env python3
"""Reproduce treeSem release metrics and produce machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
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

from treesem_adapter.bundle import ServingBundle, sha256_file
from treesem_adapter.model import ServingBundlePredictor


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1
            else probabilities < edges[index + 1])
        if selected.any():
            total += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean()))
    return total


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.reshape(-1))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "confusion_matrix": matrix.tolist(),
        "ece_10_bin": ece(labels, probabilities),
        "positive_f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": 0.5,
    }


def rebuilt_labels(raw_file: Path, bundle: ServingBundle) -> np.ndarray:
    frame = pd.read_csv(raw_file)
    labels = frame.iloc[:, -1].replace({-1: 0})
    split = bundle.manifest["split"]
    _, _, _, test_y = train_test_split(
        frame.iloc[:, 2:-4], labels,
        test_size=float(split["test_size"]), random_state=int(split["seed"]),
        stratify=labels if split["stratified"] else None)
    return np.asarray(test_y, dtype=np.int64)


def compare_metrics(artifact: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mapping = {"accuracy": "accuracy", "auc": "auc", "auprc": "auprc",
               "balanced_accuracy": "balanced_accuracy",
               "positive_f1": "positive_f1", "positive_recall": "sensitivity"}
    mismatches = [source for source, target in mapping.items()
                  if abs(float(artifact[source]) - float(actual[target])) > 1e-6]
    if np.asarray(artifact["confusion_matrix"]).tolist() != actual["confusion_matrix"]:
        mismatches.append("confusion_matrix")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cpp-dump", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("build/reports/m11-model-quality.json"))
    args = parser.parse_args()

    bundle = ServingBundle(args.bundle)
    if bundle.reference_inputs is None:
        raise SystemExit("quality audit requires a local Bundle with reference inputs")
    expected = rebuilt_labels(args.raw_file, bundle)
    if bundle.reference_labels is not None and not np.array_equal(
            expected, bundle.reference_labels):
        raise SystemExit("Bundle reference labels do not match the reconstructed split")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if sha256_file(args.artifact) != bundle.manifest["artifact_sha256"]:
        raise SystemExit("artifact checksum does not match Bundle")
    if sha256_file(args.raw_file) != bundle.manifest["raw_data_sha256"]:
        raise SystemExit("raw dataset checksum does not match Bundle")

    predictor = ServingBundlePredictor(args.bundle)
    python_results = [predictor.predict({"sample_index": index})
                      for index in range(len(bundle.reference_inputs))]
    probabilities = np.asarray([
        result["prediction"]["positive_probability"] for result in python_results])
    recomputed = metrics(expected, probabilities)
    mismatches = compare_metrics(artifact["metrics"], recomputed)

    cpp_parity: dict[str, Any] = {"status": "not_run"}
    if args.cpp_dump:
        completed = subprocess.run([str(args.cpp_dump), str(args.bundle)],
                                   check=True, capture_output=True, text=True,
                                   timeout=120)
        cpp_results = json.loads(completed.stdout)
        cpp_probabilities = np.asarray([
            result["prediction"]["positive_probability"] for result in cpp_results])
        labels_equal = all(
            left["prediction"]["label"] == right["prediction"]["label"]
            for left, right in zip(python_results, cpp_results))
        discrete_equal = labels_equal and all(
            left["prediction"]["cluster_id"] == right["prediction"]["cluster_id"] and
            left["prediction"]["tree_leaf_id"] == right["prediction"]["tree_leaf_id"]
            for left, right in zip(python_results, cpp_results))
        cpp_parity = {
            "status": "passed" if discrete_equal and
                float(np.max(np.abs(probabilities - cpp_probabilities))) <= 1e-6 else "failed",
            "discrete_results_equal": discrete_equal,
            "max_probability_delta": float(np.max(np.abs(probabilities - cpp_probabilities))),
            "metrics": metrics(expected, cpp_probabilities),
        }

    reproducible = not mismatches and cpp_parity["status"] != "failed"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "treeSem", "model_version": bundle.model_version,
        "bundle_schema_version": bundle.schema_version,
        "evidence": {
            "artifact_sha256": sha256_file(args.artifact),
            "bundle_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
            "raw_dataset_sha256": sha256_file(args.raw_file),
            "reference_rows": len(expected),
        },
        "artifact_metrics": {key: value.tolist() if hasattr(value, "tolist") else value
                             for key, value in artifact["metrics"].items()
                             if key in {"accuracy", "auc", "auprc", "balanced_accuracy",
                                        "positive_f1", "positive_recall",
                                        "positive_precision", "confusion_matrix"}},
        "pytorch_bundle_reference_metrics": recomputed,
        "artifact_metric_mismatches": mismatches,
        "cpp_onnx_parity": cpp_parity,
        "reproducible": reproducible,
        "decision": "keep_current_model" if reproducible else "investigate_before_retraining",
        "bundle_v2_required": bundle.schema_version < 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reproducible": reproducible,
                      "decision": report["decision"]}, sort_keys=True))
    if not reproducible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
