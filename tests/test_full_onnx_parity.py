#!/usr/bin/env python3
"""Compare all C++ ONNX results with the Python Bundle oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from treesem_adapter.bundle import ServingBundle
from treesem_adapter.model import ServingBundlePredictor


def path_signature(result: dict) -> list[tuple[int | None, int | None]]:
    return [
        (step.get("node_id"), step.get("leaf_id"))
        for step in result["decision_path"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-executable", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path, required=True)
    args = parser.parse_args()

    bundle = ServingBundle(args.bundle)
    predictor = ServingBundlePredictor(args.bundle)
    completed = subprocess.run(
        [str(args.dump_executable), str(args.bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    cpp_results = json.loads(completed.stdout)
    assert bundle.reference_inputs is not None
    assert len(cpp_results) == len(bundle.reference_inputs) == 1489

    max_probability_delta = 0.0
    max_confidence_delta = 0.0
    python_probabilities: list[float] = []
    cpp_probabilities: list[float] = []
    python_labels: list[int] = []
    cpp_labels: list[int] = []
    for index, cpp in enumerate(cpp_results):
        python = predictor.predict({"sample_index": index})
        py_prediction = python["prediction"]
        cpp_prediction = cpp["prediction"]
        probability_delta = abs(
            py_prediction["positive_probability"]
            - cpp_prediction["positive_probability"]
        )
        confidence_delta = abs(
            py_prediction["confidence"] - cpp_prediction["confidence"]
        )
        max_probability_delta = max(max_probability_delta, probability_delta)
        max_confidence_delta = max(max_confidence_delta, confidence_delta)
        assert probability_delta <= 1e-6
        assert confidence_delta <= 1e-6
        assert py_prediction["label"] == cpp_prediction["label"]
        assert py_prediction["cluster_id"] == cpp_prediction["cluster_id"]
        assert py_prediction["tree_leaf_id"] == cpp_prediction["tree_leaf_id"]
        assert abs(
            py_prediction["tree_probability"]
            - cpp_prediction["tree_probability"]
        ) <= 1e-12
        assert path_signature(python) == path_signature(cpp)
        assert [item["index"] for item in python["important_features"]] == [
            item["index"] for item in cpp["important_features"]
        ]
        python_probabilities.append(py_prediction["positive_probability"])
        cpp_probabilities.append(cpp_prediction["positive_probability"])
        python_labels.append(py_prediction["label"])
        cpp_labels.append(cpp_prediction["label"])

    frame = pd.read_csv(args.raw_file)
    labels = frame.iloc[:, -1].replace({-1: 0})
    split = bundle.manifest["split"]
    _, _, _, test_labels = train_test_split(
        frame.iloc[:, 2:-4],
        labels,
        test_size=float(split["test_size"]),
        random_state=int(split["seed"]),
        stratify=labels if split["stratified"] else None,
    )
    expected = np.asarray(test_labels, dtype=np.int64)
    metrics = {
        "max_probability_delta": max_probability_delta,
        "max_confidence_delta": max_confidence_delta,
        "python_accuracy": accuracy_score(expected, python_labels),
        "cpp_accuracy": accuracy_score(expected, cpp_labels),
        "python_f1": f1_score(expected, python_labels),
        "cpp_f1": f1_score(expected, cpp_labels),
        "python_auc": roc_auc_score(expected, python_probabilities),
        "cpp_auc": roc_auc_score(expected, cpp_probabilities),
    }
    assert metrics["python_accuracy"] == metrics["cpp_accuracy"]
    assert metrics["python_f1"] == metrics["cpp_f1"]
    assert abs(metrics["python_auc"] - metrics["cpp_auc"]) <= 1e-6
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
