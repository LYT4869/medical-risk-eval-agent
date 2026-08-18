from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .bundle import sha256_file


SERVING_STATE_KEYS = (
    "fc1.weight",
    "fc1.bias",
    "fc21.weight",
    "fc21.bias",
    "main.0.weight",
    "main.0.bias",
    "main.3.weight",
    "main.3.bias",
    "cluster_layer.weight",
    "cluster_layer.bias",
)
IGNORED_SHAP_STATE_KEYS = {"main.1.x", "main.1.y"}
EXPORTER_VERSION = "2.0.0"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name], dtype=np.float32), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, payload.getvalue())


def _argument(args: dict[str, Any], name: str, fallback: Any) -> Any:
    value = args.get(name)
    return fallback if value is None else value


def _tree_payload(tree: Any, selected_features: Any, input_dim: int) -> dict[str, Any]:
    import numpy as np

    selected = np.asarray(selected_features, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("artifact tree feature selection is invalid")
    if int(selected.min()) < 0 or int(selected.max()) >= input_dim:
        raise ValueError("artifact tree feature selection is out of range")
    raw = tree.tree_
    arrays = (raw.children_left, raw.children_right, raw.feature, raw.threshold, raw.value)
    if any(len(value) != raw.node_count for value in arrays):
        raise ValueError("sklearn tree arrays are inconsistent")
    nodes = []
    for node_id in range(raw.node_count):
        left = int(raw.children_left[node_id])
        right = int(raw.children_right[node_id])
        is_leaf = left == right
        value = float(np.asarray(raw.value[node_id]).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("tree contains a non-finite value")
        if is_leaf:
            nodes.append(
                {
                    "feature_index": None,
                    "id": node_id,
                    "is_leaf": True,
                    "left": None,
                    "right": None,
                    "threshold_standardized": None,
                    "value": value,
                }
            )
        else:
            local_index = int(raw.feature[node_id])
            if local_index < 0 or local_index >= len(selected):
                raise ValueError("tree local feature index is invalid")
            threshold = float(raw.threshold[node_id])
            if not np.isfinite(threshold):
                raise ValueError("tree contains a non-finite threshold")
            nodes.append(
                {
                    "feature_index": int(selected[local_index]),
                    "id": node_id,
                    "is_leaf": False,
                    "left": left,
                    "right": right,
                    "threshold_standardized": threshold,
                    "value": value,
                }
            )
    global_importance = np.zeros(input_dim, dtype=np.float64)
    local_importance = np.asarray(tree.feature_importances_, dtype=np.float64)
    if len(local_importance) != len(selected) or not np.isfinite(local_importance).all():
        raise ValueError("tree feature importance is invalid")
    for local_index, global_index in enumerate(selected):
        global_importance[int(global_index)] = float(local_importance[local_index])
    return {
        "depth": int(raw.max_depth),
        "global_feature_importance": global_importance.tolist(),
        "node_count": int(raw.node_count),
        "nodes": nodes,
        "root_node": 0,
        "selected_feature_indices": selected.tolist(),
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _index_fingerprint(values: Any) -> str:
    import numpy as np

    return _sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _artifact_metric_snapshot(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("trusted artifact does not contain a metric snapshot")
    required = ("accuracy", "auc", "auprc", "positive_f1", "balanced_accuracy",
                "positive_recall", "positive_precision", "confusion_matrix")
    if any(name not in metrics for name in required):
        raise ValueError("trusted artifact metric snapshot is incomplete")
    return {name: _jsonable(metrics[name]) for name in required}


def _expected_calibration_error(labels: Any, probabilities: Any, bins: int = 10) -> float:
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = labels.size
    result = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins + 1)[:-1],
                            np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper)
        if selected.any():
            result += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean()))
    return result


def _serving_metric_snapshot(labels: Any, probabilities: Any) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, roc_auc_score)

    expected = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    predicted = (probability >= 0.5).astype(np.int64)
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.reshape(-1))
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "auc": float(roc_auc_score(expected, probability)),
        "auprc": float(average_precision_score(expected, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "brier_score": float(np.mean(np.square(probability - expected))),
        "confusion_matrix": matrix.tolist(),
        "ece_10_bin": _expected_calibration_error(expected, probability),
        "positive_f1": float(f1_score(expected, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": 0.5,
    }


def _validate_metric_alignment(artifact: dict[str, Any], recomputed: dict[str, Any]) -> None:
    comparisons = {
        "accuracy": "accuracy",
        "auc": "auc",
        "auprc": "auprc",
        "positive_f1": "positive_f1",
        "balanced_accuracy": "balanced_accuracy",
        "positive_recall": "sensitivity",
    }
    mismatches = []
    for artifact_name, recomputed_name in comparisons.items():
        if abs(float(artifact[artifact_name]) - float(recomputed[recomputed_name])) > 1e-6:
            mismatches.append(artifact_name)
    if _jsonable(artifact["confusion_matrix"]) != recomputed["confusion_matrix"]:
        mismatches.append("confusion_matrix")
    if mismatches:
        raise ValueError("serving predictions do not reproduce artifact metrics: " +
                         ", ".join(sorted(mismatches)))


def _runtime_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn
    import torch

    return {
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    }


def export_bundle(
    *,
    artifact_path: str | Path,
    raw_file: str | Path,
    output_dir: str | Path,
    include_reference_dataset: bool = False,
    force: bool = False,
    training_code_commit: str | None = None,
) -> Path:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    artifact_path = Path(artifact_path).resolve()
    raw_file = Path(raw_file).resolve()
    output_dir = Path(output_dir).resolve()
    if not artifact_path.is_file() or not raw_file.is_file():
        raise FileNotFoundError("artifact and raw CSV must exist")
    artifact_sha = sha256_file(artifact_path)
    raw_sha = sha256_file(raw_file)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    required = {"model_state", "model_config", "tree", "args"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("trusted artifact is missing serving fields")
    selected = artifact.get("t_star", artifact.get("u_star"))
    if selected is None:
        raise ValueError("trusted artifact has no tree feature selection")

    input_dim = int(artifact["model_config"]["input_dim"])
    if input_dim != 49:
        raise ValueError("only the 49-dimensional PPH model is supported")
    state = artifact["model_state"]
    allowed_state = set(SERVING_STATE_KEYS) | IGNORED_SHAP_STATE_KEYS | {
        "fc22.weight", "fc22.bias", "fc3.weight", "fc3.bias", "fc4.weight", "fc4.bias"
    }
    unknown = sorted(set(state).difference(allowed_state))
    missing = sorted(set(SERVING_STATE_KEYS).difference(state))
    if unknown or missing:
        raise ValueError(f"artifact state mismatch; missing={missing}, unknown={unknown}")
    arrays = {name: state[name].detach().cpu().numpy().astype(np.float32) for name in SERVING_STATE_KEYS}
    if int(arrays["main.3.weight"].shape[0]) != 2:
        raise ValueError("only binary classification is supported")
    cluster_count = int(arrays["cluster_layer.weight"].shape[0])

    frame = pd.read_csv(raw_file)
    features = frame.iloc[:, 2:-4]
    labels = frame.iloc[:, -1].replace({-1: 0})
    if features.shape[1] != input_dim:
        raise ValueError("raw CSV does not contain the expected 49 PPH features")
    run_args = dict(artifact.get("args") or {})
    seed = int(_argument(run_args, "seed", 42))
    test_size = float(_argument(run_args, "test_size", 0.2))
    stratify = labels if bool(_argument(run_args, "stratify_split", False)) else None
    train_x, test_x, _train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    scaler = StandardScaler().fit(train_x)
    reference_inputs = scaler.transform(test_x).astype("<f4")
    if not np.isfinite(reference_inputs).all() or reference_inputs.shape[1] != input_dim:
        raise ValueError("reconstructed reference data is invalid")

    model_version = f"pph-seed{seed}-{artifact_sha[:12]}"
    final_directory = output_dir / model_version
    if final_directory.exists() and not force:
        raise FileExistsError(f"bundle already exists: {final_directory}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_version}.", dir=output_dir))
    try:
        feature_rows = [
            {
                "display_name": str(name).replace("_", " "),
                "index": index,
                "name": str(name),
                "source_dtype": str(features.dtypes.iloc[index]),
                "unit": None,
                "value_kind": "numeric",
            }
            for index, name in enumerate(features.columns)
        ]
        _write_json(temporary / "feature_schema.json", {"features": feature_rows})
        _write_json(
            temporary / "preprocessing.json",
            {
                "formula": "standardized=(original-mean)/scale",
                "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
                "method": "StandardScaler",
                "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
            },
        )
        _write_deterministic_npz(temporary / "neural_state.npz", arrays)
        _write_json(temporary / "tree.json", _tree_payload(artifact["tree"], selected, input_dim))
        if include_reference_dataset:
            (temporary / "reference_inputs.f32").write_bytes(reference_inputs.tobytes(order="C"))
            (temporary / "reference_labels.i8").write_bytes(
                np.asarray(test_y, dtype="i1").tobytes(order="C"))

        # Golden cases intentionally contain only deterministic model facts.
        negative = np.flatnonzero(np.asarray(test_y) == 0)[:16]
        positive = np.flatnonzero(np.asarray(test_y) == 1)[:16]
        selected_cases = sorted(int(value) for value in np.concatenate((negative, positive)))
        _write_json(temporary / "reference_outputs.json", {"case_indices": selected_cases})

        files = {}
        logical_names = {
            "feature_schema": "feature_schema.json",
            "preprocessing": "preprocessing.json",
            "neural_state": "neural_state.npz",
            "tree": "tree.json",
            "reference_outputs": "reference_outputs.json",
        }
        if include_reference_dataset:
            logical_names["reference_inputs"] = "reference_inputs.f32"
            logical_names["reference_labels"] = "reference_labels.i8"
        for logical_name, filename in logical_names.items():
            files[logical_name] = _file_metadata(temporary / filename)
        manifest = {
            "artifact_sha256": artifact_sha,
            "bundle_schema_version": 2,
            "class_count": 2,
            "cluster_count": cluster_count,
            "contains_reference_dataset": bool(include_reference_dataset),
            "contains_reference_labels": bool(include_reference_dataset),
            "dataset": "pph",
            "files": files,
            "input_dim": input_dim,
            "model": "treeSem",
            "model_version": model_version,
            "onnx": {
                "input_name": "preprocessed_features",
                "opset": 17,
                "output_names": ["class_probabilities", "cluster_logits"],
                "present": False,
            },
            "raw_data_sha256": raw_sha,
            "reference_rows": int(reference_inputs.shape[0]) if include_reference_dataset else 0,
            "seed": seed,
            "split": {
                "seed": seed,
                "stratified": stratify is not None,
                "test_size": test_size,
            },
            "provenance": {
                "artifact_sha256": artifact_sha,
                "exporter_version": EXPORTER_VERSION,
                "feature_schema_sha256": sha256_file(temporary / "feature_schema.json"),
                "label_schema": {"negative": 0, "positive": 1,
                                 "source_negative_value": -1, "threshold": 0.5},
                "preprocessing_sha256": sha256_file(temporary / "preprocessing.json"),
                "raw_dataset_sha256": raw_sha,
                "runtime_versions": _runtime_versions(),
                "split_fingerprint": {
                    "combined_sha256": _sha256_bytes(
                        bytes.fromhex(_index_fingerprint(train_x.index)) +
                        bytes.fromhex(_index_fingerprint(test_x.index))),
                    "test_indices_sha256": _index_fingerprint(test_x.index),
                    "train_indices_sha256": _index_fingerprint(train_x.index),
                },
                "training_code_commit": training_code_commit or
                    os.environ.get("TREESEM_TRAINING_CODE_COMMIT") or None,
            },
            "evaluation": {
                "artifact_metrics": _artifact_metric_snapshot(artifact.get("metrics")),
                # A temporary validated snapshot lets the immutable Bundle loader
                # construct the oracle used below. It is replaced with full
                # serving recomputation before the temporary directory is promoted.
                "recomputed_metrics": _artifact_metric_snapshot(artifact.get("metrics")),
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        from .model import ServingBundlePredictor

        predictor = ServingBundlePredictor(temporary)
        probabilities = [
            predictor.predict({"preprocessed_features": row.astype(float).tolist()})[
                "prediction"]["positive_probability"]
            for row in reference_inputs
        ]
        recomputed_metrics = _serving_metric_snapshot(test_y, probabilities)
        _validate_metric_alignment(manifest["evaluation"]["artifact_metrics"],
                                   recomputed_metrics)
        manifest["evaluation"]["recomputed_metrics"] = recomputed_metrics
        golden_cases = [
            {
                "reference_index": index,
                "result": predictor.predict(
                    {
                        "preprocessed_features": reference_inputs[index]
                        .astype(float)
                        .tolist()
                    }
                ),
            }
            for index in selected_cases
        ]
        _write_json(
            temporary / "reference_outputs.json",
            {"cases": golden_cases},
        )
        manifest["files"]["reference_outputs"] = _file_metadata(
            temporary / "reference_outputs.json"
        )
        _write_json(temporary / "manifest.json", manifest)
        if final_directory.exists():
            shutil.rmtree(final_directory)
        temporary.rename(final_directory)
        return final_directory
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trusted treeSem PPH serving bundle")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-reference-dataset", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--training-code-commit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = export_bundle(
        artifact_path=args.artifact,
        raw_file=args.raw_file,
        output_dir=args.output_dir,
        include_reference_dataset=args.include_reference_dataset,
        force=args.force,
        training_code_commit=args.training_code_commit,
    )
    print(path)


if __name__ == "__main__":
    main()
