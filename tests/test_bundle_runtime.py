#!/usr/bin/env python3
"""Validate a real treeSem serving bundle and its deterministic exporter."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from treesem_adapter.bundle import (
    BundleValidationError,
    FeaturePreprocessor,
    ServingBundle,
    sha256_file,
)
from treesem_adapter.export_bundle import export_bundle
from treesem_adapter.model import RequestValidationError, ServingBundlePredictor


def _business_files(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in directory.iterdir()
        if path.is_file()
    }


def _assert_close_result(left: dict, right: dict) -> None:
    left_prediction = left["prediction"]
    right_prediction = right["prediction"]
    for key in ("label", "cluster_id", "tree_leaf_id"):
        assert left_prediction[key] == right_prediction[key], key
    for key in ("positive_probability", "confidence", "tree_probability"):
        assert abs(left_prediction[key] - right_prediction[key]) <= 1e-6, key
    assert [step.get("node_id", step.get("leaf_id")) for step in left["decision_path"]] == [
        step.get("node_id", step.get("leaf_id")) for step in right["decision_path"]
    ]


def validate_bundle(bundle_directory: Path) -> None:
    bundle = ServingBundle(bundle_directory)
    assert bundle.reference_inputs is not None
    assert bundle.reference_inputs.shape == (1489, 49)
    predictor = ServingBundlePredictor(bundle_directory)
    preprocessor = FeaturePreprocessor(bundle)

    standardized = np.asarray(bundle.reference_inputs[0], dtype=np.float32)
    raw = {
        feature.name: preprocessor.original_value(feature.index, standardized[feature.index])
        for feature in bundle.features
    }
    reconstructed = preprocessor.raw_object_to_standardized(raw)
    assert float(np.max(np.abs(reconstructed - standardized))) <= 1e-6
    restored = np.asarray(
        [
            preprocessor.original_value(index, reconstructed[index])
            for index in range(bundle.input_dim)
        ],
        dtype=np.float64,
    )
    original = np.asarray([raw[name] for name in bundle.feature_names], dtype=np.float64)
    assert float(np.max(np.abs(restored - original))) <= 1e-5

    by_index = predictor.predict({"sample_index": 0})
    by_preprocessed = predictor.predict(
        {"preprocessed_features": standardized.astype(float).tolist()}
    )
    by_raw = predictor.predict({"raw_features": raw})
    _assert_close_result(by_index, by_preprocessed)
    _assert_close_result(by_index, by_raw)

    invalid_raw = dict(raw)
    invalid_raw.pop(bundle.feature_names[0])
    try:
        predictor.predict({"raw_features": invalid_raw})
    except RequestValidationError:
        pass
    else:
        raise AssertionError("missing raw feature was accepted")

    with tempfile.TemporaryDirectory(prefix="treesem-corrupt-") as temporary:
        copied = Path(temporary) / "bundle"
        shutil.copytree(bundle_directory, copied)
        with (copied / "tree.json").open("ab") as output:
            output.write(b"\n")
        try:
            ServingBundle(copied)
        except BundleValidationError:
            pass
        else:
            raise AssertionError("modified bundle file passed checksum validation")


def validate_deterministic_export(artifact: Path, raw_file: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="treesem-export-a-") as first_root, tempfile.TemporaryDirectory(
        prefix="treesem-export-b-"
    ) as second_root:
        first = export_bundle(
            artifact_path=artifact,
            raw_file=raw_file,
            output_dir=first_root,
            include_reference_dataset=True,
        )
        second = export_bundle(
            artifact_path=artifact,
            raw_file=raw_file,
            output_dir=second_root,
            include_reference_dataset=True,
        )
        assert first.name == second.name
        assert _business_files(first) == _business_files(second)
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["bundle_schema_version"] == 2
        assert manifest["model_version"] == first.name
        assert manifest["contains_reference_dataset"] is True
        assert manifest["contains_reference_labels"] is True
        assert manifest["reference_rows"] == 1489
        assert manifest["evaluation"]["recomputed_metrics"]["accuracy"] > 0.9
        loaded = ServingBundle(first)
        assert loaded.reference_labels is not None
        assert loaded.reference_labels.shape == (1489,)
        second_manifest = json.loads(
            (second / "manifest.json").read_text(encoding="utf-8"))
        second_manifest["provenance"]["artifact_sha256"] = "0" * 64
        (second / "manifest.json").write_text(
            json.dumps(second_manifest, sort_keys=True), encoding="utf-8")
        try:
            ServingBundle(second)
        except BundleValidationError:
            pass
        else:
            raise AssertionError("inconsistent v2 provenance was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path, required=True)
    args = parser.parse_args()
    validate_bundle(args.bundle.resolve())
    validate_deterministic_export(args.artifact.resolve(), args.raw_file.resolve())
    print("treeSem serving bundle validation passed")


if __name__ == "__main__":
    main()
