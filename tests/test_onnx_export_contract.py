#!/usr/bin/env python3
"""Validate ONNX export determinism and production Bundle separation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from treesem_adapter.bundle import ServingBundle, sha256_file
from treesem_adapter.export_bundle import export_bundle
from treesem_adapter.export_onnx import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-bundle", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path, required=True)
    args = parser.parse_args()

    validation = ServingBundle(args.validation_bundle)
    onnx_path = validation.directory / "model.onnx"
    initial_sha = sha256_file(onnx_path)
    export_onnx(validation.directory, force=True)
    assert sha256_file(onnx_path) == initial_sha

    with tempfile.TemporaryDirectory(prefix="treesem-production-") as temporary:
        production_path = export_bundle(
            artifact_path=args.artifact,
            raw_file=args.raw_file,
            output_dir=temporary,
            include_reference_dataset=False,
        )
        export_onnx(
            production_path,
            validation_bundle_directory=validation.directory,
        )
        production = ServingBundle(production_path)
        assert production.model_version == validation.model_version
        assert production.reference_inputs is None
        assert production.manifest["contains_reference_dataset"] is False
        assert production.manifest["onnx"]["present"] is True
        assert (production.directory / "model.onnx").is_file()

    print("treeSem ONNX export contract passed")


if __name__ == "__main__":
    main()
