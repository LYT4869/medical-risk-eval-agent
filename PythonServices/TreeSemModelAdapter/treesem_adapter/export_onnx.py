from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .bundle import ServingBundle, sha256_file


INPUT_NAME = "preprocessed_features"
OUTPUT_NAMES = ("class_probabilities", "cluster_logits")
OPSET_VERSION = 17


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}


def export_onnx(
    bundle_directory: str | Path,
    *,
    validation_bundle_directory: str | Path | None = None,
    force: bool = False,
) -> dict[str, float]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    bundle = ServingBundle(bundle_directory)
    validation_bundle = (
        bundle
        if validation_bundle_directory is None
        else ServingBundle(validation_bundle_directory)
    )
    if validation_bundle.model_version != bundle.model_version:
        raise ValueError("validation bundle model_version does not match target")
    if validation_bundle.reference_inputs is None:
        raise ValueError(
            "ONNX export requires reference inputs in the target or validation bundle"
        )
    destination = bundle.directory / "model.onnx"
    temporary = bundle.directory / ".model.onnx.tmp"
    if destination.exists() and not force:
        raise FileExistsError("model.onnx already exists; pass --force to replace it")

    class DeterministicTreeSem(torch.nn.Module):
        def __init__(self, state: dict[str, Any]) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(49, 64)
            self.fc21 = torch.nn.Linear(64, 64)
            self.classifier_hidden = torch.nn.Linear(64, 32)
            self.classifier_output = torch.nn.Linear(32, 2)
            self.cluster = torch.nn.Linear(64, bundle.cluster_count)
            mappings = (
                (self.fc1, "fc1"),
                (self.fc21, "fc21"),
                (self.classifier_hidden, "main.0"),
                (self.classifier_output, "main.3"),
                (self.cluster, "cluster_layer"),
            )
            with torch.no_grad():
                for layer, prefix in mappings:
                    layer.weight.copy_(torch.from_numpy(state[f"{prefix}.weight"]))
                    layer.bias.copy_(torch.from_numpy(state[f"{prefix}.bias"]))

        def forward(self, features: Any) -> tuple[Any, Any]:
            hidden = torch.relu(self.fc1(features))
            mu = self.fc21(hidden)
            logits = self.classifier_output(
                torch.relu(self.classifier_hidden(mu))
            )
            return torch.softmax(logits, dim=1), self.cluster(mu)

    model = DeterministicTreeSem(bundle.neural_state).eval()
    reference = np.asarray(validation_bundle.reference_inputs, dtype=np.float32)
    try:
        torch.onnx.export(
            model,
            torch.from_numpy(reference[:1]),
            temporary,
            export_params=True,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                INPUT_NAME: {0: "batch"},
                OUTPUT_NAMES[0]: {0: "batch"},
                OUTPUT_NAMES[1]: {0: "batch"},
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
        checked = onnx.load(temporary)
        onnx.checker.check_model(checked)
        inferred = onnx.shape_inference.infer_shapes(checked)
        onnx.checker.check_model(inferred)
        onnx.save(inferred, temporary)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(temporary),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        assert [value.name for value in session.get_inputs()] == [INPUT_NAME]
        assert [value.name for value in session.get_outputs()] == list(OUTPUT_NAMES)
        with torch.inference_mode():
            torch_probabilities, torch_clusters = model(torch.from_numpy(reference))
        ort_probabilities, ort_clusters = session.run(
            list(OUTPUT_NAMES), {INPUT_NAME: reference}
        )
        probability_delta = float(
            np.max(np.abs(ort_probabilities - torch_probabilities.numpy()))
        )
        cluster_delta = float(np.max(np.abs(ort_clusters - torch_clusters.numpy())))
        if probability_delta > 1e-5 or cluster_delta > 1e-5:
            raise RuntimeError(
                "ONNX parity failed: "
                f"probability_delta={probability_delta}, cluster_delta={cluster_delta}"
            )
        if not np.array_equal(
            np.argmax(ort_probabilities, axis=1),
            np.argmax(torch_probabilities.numpy(), axis=1),
        ):
            raise RuntimeError("ONNX export changed a class label")
        if not np.array_equal(
            np.argmax(ort_clusters, axis=1),
            np.argmax(torch_clusters.numpy(), axis=1),
        ):
            raise RuntimeError("ONNX export changed a cluster id")

        os.replace(temporary, destination)
        manifest_path = bundle.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["model_onnx"] = _file_metadata(destination)
        manifest["onnx"] = {
            "input_name": INPUT_NAME,
            "input_shape": ["batch", bundle.input_dim],
            "opset": OPSET_VERSION,
            "output_names": list(OUTPUT_NAMES),
            "output_shapes": [["batch", bundle.class_count], ["batch", bundle.cluster_count]],
            "present": True,
            "runtime_version": "1.20.1",
        }
        _write_json(manifest_path, manifest)
        ServingBundle(bundle.directory)
        return {
            "max_probability_delta": probability_delta,
            "max_cluster_logit_delta": cluster_delta,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export treeSem deterministic ONNX graph")
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--validation-bundle",
        help="local same-version Bundle containing reference_inputs.f32",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metrics = export_onnx(
        args.bundle,
        validation_bundle_directory=args.validation_bundle,
        force=args.force,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
