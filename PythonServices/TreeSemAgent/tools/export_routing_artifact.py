from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.routing import RuleRouter, SafetyGate
from agent.routing_artifact import (
    canonical_intent_examples,
    intent_examples_sha256,
)
from agent.semantic_routing import RoutingThresholds, SemanticScorer
from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry
from evaluation.run_routing_evaluation import RoutingCase, load_cases


EXPORTER_VERSION = "1"
SUPPORTED_MODEL = "intfloat/multilingual-e5-small"
SUPPORTED_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SUPPORTED_BACKENDS = frozenset({"onnx_fp32", "onnx_int8"})


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceModelContract:
    embedding_dimension: int
    max_sequence_length: int
    pad_token: str
    pad_token_id: int
    input_names: tuple[str, ...]


@dataclass(frozen=True)
class ExportSettings:
    source_model_id: str
    source_model_revision: str
    backend: str
    tasks_path: Path
    parity_cases_path: Path
    thresholds_path: Path
    output_root: Path
    opset: int = 17


def stable_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExportError("artifact metadata must contain finite JSON values") \
            from exc
    return (rendered + "\n").encode("utf-8")


def artifact_version(payload: Mapping[str, object]) -> str:
    return "routing-" + hashlib.sha256(stable_json_bytes(payload)).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_model(model: Any) -> SourceModelContract:
    try:
        modules = [model[index] for index in range(len(model))]
    except (TypeError, IndexError, AttributeError) as exc:
        raise ExportError("cannot inspect Sentence Transformer module chain") \
            from exc
    if [type(module).__name__ for module in modules] != [
            "Transformer", "Pooling", "Normalize"]:
        raise ExportError("unsupported Sentence Transformer module chain")
    pooling = modules[1]
    if (getattr(pooling, "pooling_mode", None) != "mean" or
            getattr(pooling, "include_prompt", None) is not True):
        raise ExportError("source model must use prompt-inclusive mean pooling")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ExportError("source model tokenizer is unavailable")
    input_names = tuple(getattr(tokenizer, "model_input_names", ()))
    if (getattr(tokenizer, "padding_side", None) != "right" or
            getattr(tokenizer, "truncation_side", None) != "right" or
            getattr(tokenizer, "pad_token", None) != "<pad>" or
            getattr(tokenizer, "pad_token_id", None) != 1 or
            input_names != ("input_ids", "attention_mask")):
        raise ExportError("unsupported source tokenizer contract")
    max_length = getattr(model, "max_seq_length", None)
    if max_length != 512:
        raise ExportError("unsupported source tokenizer maximum length")
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if not callable(dimension_getter):
        dimension_getter = getattr(
            model, "get_sentence_embedding_dimension", None)
    dimension = dimension_getter() if callable(dimension_getter) else 384
    if isinstance(dimension, bool) or dimension != 384:
        raise ExportError("unsupported source embedding dimension")
    return SourceModelContract(
        embedding_dimension=dimension,
        max_sequence_length=max_length,
        pad_token=tokenizer.pad_token,
        pad_token_id=tokenizer.pad_token_id,
        input_names=input_names,
    )


def _matrix(value: Any) -> list[list[float]]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    try:
        return [[float(item) for item in row] for row in raw]
    except (TypeError, ValueError) as exc:
        raise ExportError("embedding encoder returned an invalid matrix") from exc


def write_intent_embeddings(
        registry: TaskRegistry,
        encoder: Callable[[Sequence[str]], Any], output: Path, *,
        passage_prefix: str, embedding_dimension: int) -> str:
    examples = canonical_intent_examples(registry)
    rows = _matrix(encoder([passage_prefix + item for item in examples]))
    if len(rows) != len(examples):
        raise ExportError("intent embedding count mismatch")
    encoded = bytearray()
    for row in rows:
        if len(row) != embedding_dimension:
            raise ExportError("intent embedding dimension mismatch")
        if any(not math.isfinite(value) for value in row):
            raise ExportError("intent embeddings must contain finite values")
        norm = math.sqrt(sum(value * value for value in row))
        if not math.isfinite(norm) or norm <= 0:
            raise ExportError("intent embeddings must have positive norm")
        if abs(norm - 1.0) > 1e-3:
            raise ExportError("intent embeddings must be L2 normalized")
        for value in row:
            encoded.extend(struct.pack("<f", value))
    output.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


class _SourceEmbeddingProvider:
    def __init__(self, model: Any):
        self._model = model

    def encode_examples(self, texts: Sequence[str]) -> list[list[float]]:
        return _matrix(self._model.encode(
            ["passage: " + item for item in texts],
            normalize_embeddings=True, convert_to_numpy=True))

    def encode_query(self, text: str) -> list[float]:
        return _matrix(self._model.encode(
            ["query: " + text], normalize_embeddings=True,
            convert_to_numpy=True))[0]


def _load_thresholds(path: Path, settings: ExportSettings) -> RoutingThresholds:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("schema_version") != 1 or
                payload.get("model") != settings.source_model_id or
                payload.get("revision") != settings.source_model_revision):
            raise ExportError("routing thresholds do not match source model")
        return RoutingThresholds(**payload["thresholds"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError,
            ValueError) as exc:
        if isinstance(exc, ExportError):
            raise
        raise ExportError("cannot load routing thresholds") from exc


def _golden_scope(case: RoutingCase, scorer: SemanticScorer,
                  rules: RuleRouter, safety: SafetyGate) -> str:
    safety_decision = safety.evaluate(case.message)
    if not safety_decision.allowed:
        return (safety_decision.refusal_scope.value
                if safety_decision.refusal_scope is not None else "unknown")
    rule = rules.route(case.message)
    if rule is not None:
        return rule.scope.value
    return scorer.route(case.message).scope.value


def _write_golden_routes(
        path: Path, cases: list[RoutingCase], registry: TaskRegistry,
        source_model: Any, thresholds: RoutingThresholds) -> str:
    if len(cases) != 150:
        raise ExportError("frozen routing parity set must contain 150 cases")
    scorer = SemanticScorer(
        registry, _SourceEmbeddingProvider(source_model), thresholds)
    rules = RuleRouter(registry)
    safety = SafetyGate()
    rows = []
    for case in cases:
        rows.append({
            "case_id": case.case_id,
            "expected_scope": case.expected_scope,
            "golden_scope": _golden_scope(case, scorer, rules, safety),
            "message_sha256": hashlib.sha256(
                case.message.encode("utf-8")).hexdigest(),
        })
    payload = {"schema_version": 1, "case_count": len(rows), "cases": rows}
    path.write_bytes(stable_json_bytes(payload))
    return _sha256(path)


def _export_fp32_onnx(source_model: Any, destination: Path,
                      contract: SourceModelContract, opset: int) -> None:
    try:
        import onnx
        import torch
    except ModuleNotFoundError as exc:
        raise ExportError("ONNX export dependencies are unavailable") from exc

    class E5SentenceEmbeddingModule(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            token_embeddings = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(
                dim=1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    source_model.eval()
    encoded = source_model.tokenizer(
        ["query: routing export parity"], padding=True, truncation=True,
        max_length=contract.max_sequence_length, return_tensors="pt")
    wrapper = E5SentenceEmbeddingModule(source_model[0].auto_model)
    wrapper.eval()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=torch.jit.TracerWarning,
            message="torch.tensor results are registered as constants.*")
        torch.onnx.export(
            wrapper,
            (encoded["input_ids"], encoded["attention_mask"]),
            destination,
            input_names=list(contract.input_names),
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(str(destination))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    onnx.save_model(inferred, str(destination), save_as_external_data=False)


def quantize_dynamic_int8(source: Path, destination: Path) -> None:
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ModuleNotFoundError as exc:
        raise ExportError("ONNX quantization dependencies are unavailable") \
            from exc
    quantize_dynamic(
        model_input=str(source), model_output=str(destination),
        per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"])
    model = onnx.load(str(destination))
    onnx.checker.check_model(model)


def _onnx_encoder(model_path: Path, tokenizer_path: Path,
                  contract: SourceModelContract) -> Callable[[Sequence[str]], Any]:
    try:
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise ExportError("ONNX runtime dependencies are unavailable") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(
        max_length=contract.max_sequence_length, direction="right")
    tokenizer.enable_padding(
        direction="right", pad_id=contract.pad_token_id,
        pad_token=contract.pad_token)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        str(model_path), sess_options=options,
        providers=["CPUExecutionProvider"])
    if ([item.name for item in session.get_inputs()] !=
            list(contract.input_names) or
            [item.name for item in session.get_outputs()] !=
            ["sentence_embedding"]):
        raise ExportError("exported ONNX I/O contract mismatch")

    def encode(texts: Sequence[str]):
        encoded = tokenizer.encode_batch(list(texts), add_special_tokens=True)
        feeds = {
            "input_ids": numpy.asarray(
                [item.ids for item in encoded], dtype=numpy.int64),
            "attention_mask": numpy.asarray(
                [item.attention_mask for item in encoded], dtype=numpy.int64),
        }
        return session.run(["sentence_embedding"], feeds)[0]

    return encode


def _load_source_model(settings: ExportSettings) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ExportError("Sentence Transformers export dependency is unavailable") \
            from exc
    return SentenceTransformer(
        settings.source_model_id,
        revision=settings.source_model_revision,
        local_files_only=True)


def _validate_settings(settings: ExportSettings) -> None:
    if settings.source_model_id != SUPPORTED_MODEL:
        raise ExportError("unsupported routing source model")
    if settings.source_model_revision != SUPPORTED_REVISION:
        raise ExportError("unsupported routing source revision")
    if settings.backend not in SUPPORTED_BACKENDS:
        raise ExportError("unsupported routing embedding backend")
    if settings.opset != 17:
        raise ExportError("unsupported routing ONNX opset")
    for path in (settings.tasks_path, settings.parity_cases_path,
                 settings.thresholds_path):
        if path.is_symlink() or not path.is_file():
            raise ExportError("routing export input is missing or unsafe")


def export_routing_artifact(settings: ExportSettings) -> Path:
    _validate_settings(settings)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    registry = TaskRegistry.load(
        settings.tasks_path, SUPPORTED_DOMAIN_TOOLS)
    cases = load_cases(settings.parity_cases_path)
    thresholds = _load_thresholds(settings.thresholds_path, settings)
    source_model = _load_source_model(settings)
    contract = validate_source_model(source_model)

    settings.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".routing-export-", dir=settings.output_root) as temporary:
        staging = Path(temporary)
        fp32_path = staging / "model.fp32.onnx"
        selected_model_path = staging / "model.onnx"
        tokenizer_path = staging / "tokenizer.json"
        tokenizer_config_path = staging / "tokenizer_config.json"
        intent_path = staging / "intent_embeddings.f32"
        golden_path = staging / "golden_routes.json"

        _export_fp32_onnx(
            source_model, fp32_path, contract, settings.opset)
        if settings.backend == "onnx_int8":
            quantize_dynamic_int8(fp32_path, selected_model_path)
            fp32_path.unlink()
        else:
            fp32_path.replace(selected_model_path)

        source_model.tokenizer.backend_tokenizer.save(str(tokenizer_path))
        tokenizer_config_path.write_bytes(stable_json_bytes({
            "max_sequence_length": contract.max_sequence_length,
            "padding_side": "right",
            "truncation_side": "right",
            "pad_token": contract.pad_token,
            "pad_token_id": contract.pad_token_id,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        }))
        selected_encoder = _onnx_encoder(
            selected_model_path, tokenizer_path, contract)
        write_intent_embeddings(
            registry, selected_encoder, intent_path,
            passage_prefix="passage: ",
            embedding_dimension=contract.embedding_dimension)
        _write_golden_routes(
            golden_path, cases, registry, source_model, thresholds)

        try:
            import tokenizers
        except ModuleNotFoundError as exc:
            raise ExportError("tokenizers export dependency is unavailable") \
                from exc
        checksums = {
            "model_sha256": _sha256(selected_model_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "tokenizer_config_sha256": _sha256(tokenizer_config_path),
            "intent_embeddings_sha256": _sha256(intent_path),
            "golden_routes_sha256": _sha256(golden_path),
        }
        version_input = {
            "source_model_id": settings.source_model_id,
            "source_model_revision": settings.source_model_revision,
            "embedding_backend": settings.backend,
            "onnx_opset": settings.opset,
            "task_registry_sha256": _sha256(settings.tasks_path),
            "intent_examples_sha256": intent_examples_sha256(
                canonical_intent_examples(registry)),
            "exporter_version": EXPORTER_VERSION,
            **checksums,
        }
        version = artifact_version(version_input)
        manifest = {
            "schema_version": 1,
            "artifact_version": version,
            "source_model_id": settings.source_model_id,
            "source_model_revision": settings.source_model_revision,
            "embedding_backend": settings.backend,
            "onnx_opset": settings.opset,
            "onnx_input_names": list(contract.input_names),
            "onnx_output_name": "sentence_embedding",
            "embedding_dimension": contract.embedding_dimension,
            "intent_example_count": len(canonical_intent_examples(registry)),
            "intent_examples_sha256": version_input[
                "intent_examples_sha256"],
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "max_sequence_length": contract.max_sequence_length,
            "pooling": "attention_mask_mean",
            "normalization": "l2",
            "pad_token": contract.pad_token,
            "pad_token_id": contract.pad_token_id,
            "tokenizer_library_version": tokenizers.__version__,
            "exporter_version": EXPORTER_VERSION,
            "task_registry_sha256": version_input["task_registry_sha256"],
            **checksums,
        }
        (staging / "manifest.json").write_bytes(stable_json_bytes(manifest))

        destination = settings.output_root / version
        if destination.exists():
            raise ExportError("routing artifact version already exists")
        staging.rename(destination)
        return destination


def _settings_from_arguments() -> ExportSettings:
    parser = argparse.ArgumentParser(
        description="Export a versioned treeSem Agent routing artifact")
    parser.add_argument("--backend", choices=tuple(sorted(SUPPORTED_BACKENDS)),
                        required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--parity-cases", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    return ExportSettings(
        source_model_id=arguments.model,
        source_model_revision=arguments.revision,
        backend=arguments.backend,
        tasks_path=arguments.tasks,
        parity_cases_path=arguments.parity_cases,
        thresholds_path=arguments.thresholds,
        output_root=arguments.output_root)


def main() -> int:
    destination = export_routing_artifact(_settings_from_arguments())
    print(json.dumps({
        "status": "ok",
        "artifact_dir": str(destination),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
