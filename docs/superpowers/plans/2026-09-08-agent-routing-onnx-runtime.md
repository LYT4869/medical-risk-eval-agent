# Agent Routing ONNX Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the production Agent's PyTorch/Sentence Transformers semantic-routing runtime with a checksum-verified FP32/INT8 ONNX artifact while preserving the existing rule-first policy, bounded execution, and safe degradation behavior.

**Architecture:** A deterministic offline exporter reproduces the pinned Sentence Transformers Transformer/mean-pooling/normalization chain and packages an ONNX model, tokenizer, and backend-aligned intent embeddings. At runtime, an `OnnxEmbeddingProvider` loads that immutable artifact once and remains behind the existing `SemanticRoutingExecutor`; routing thresholds and Workflow selection remain unchanged. FP32 ONNX must pass the frozen 150-case parity gate before INT8 is evaluated, and execution stops before any new quality corpus is generated.

**Tech Stack:** Python 3.10, PyTorch 2.5.1 CPU and Sentence Transformers 5.7.0 for offline golden/export only, ONNX 1.17.0, ONNX Runtime 1.20.1 CPU, tokenizers 0.22.2, NumPy 2.0.1, unittest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-08-agent-routing-onnx-runtime-design.md`

## Global Constraints

- Do not change Safety Gate, Rule Router, Task Registry intent definitions, Workflow stages, Tool permissions, or medical response policy.
- Keep `TREESEM_AGENT_ROUTING_MODE=rule` as the default.
- Keep semantic inference behind the existing fixed-size executor, bounded admission, and timeout.
- Use `intfloat/multilingual-e5-small` at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Reproduce the current 512-token, right-padding, right-truncation, `query: ` / `passage: `, attention-mask mean-pooling, L2-normalization behavior.
- Query and intent embeddings in one deployed artifact must come from the same ONNX backend and model checksum.
- Runtime startup is offline; generated artifacts are ignored by Git and mounted read-only.
- FP32 route parity across the frozen 150 cases is a hard prerequisite for INT8 evaluation.
- INT8 is promoted only when it introduces no frozen-set route regression and passes all safety/resource gates; otherwise use FP32 ONNX.
- Do not call an LLM to generate the 300-400-case Routing Quality Set. Stop after Task 8 and ask the user to arrange generation.
- Do not push, merge, or create a PR. Create local checkpoint commits only.

## File and Responsibility Map

**Create:**

- `PythonServices/TreeSemAgent/agent/routing_artifact.py` — strict manifest, checksum, matrix, and Task Registry validation.
- `PythonServices/TreeSemAgent/agent/onnx_embedding_provider.py` — tokenizer and ONNX Runtime inference behind the existing embedding protocol.
- `PythonServices/TreeSemAgent/tools/__init__.py` — marks the offline export tooling package.
- `PythonServices/TreeSemAgent/tools/export_routing_artifact.py` — deterministic FP32 export, optional INT8 quantization, backend-aligned intent embeddings, and manifest generation.
- `PythonServices/TreeSemAgent/evaluation/compare_routing_backends.py` — frozen-set embedding/score/route parity report.
- `PythonServices/TreeSemAgent/evaluation/benchmark_routing_runtime.py` — cold start, warmed latency, and process RSS measurement.
- `PythonServices/TreeSemAgent/requirements-routing-export.txt` — offline heavy export dependencies.
- `PythonServices/TreeSemAgent/tests/test_routing_artifact.py` — artifact fail-fast and binary matrix tests.
- `PythonServices/TreeSemAgent/tests/test_onnx_embedding_provider.py` — tokenization, input tensor, output, and precomputed-example tests.
- `PythonServices/TreeSemAgent/tests/test_routing_export.py` — deterministic manifest/export helper tests with fake model backends.
- `PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py` — parity report behavior and gate tests.
- `PythonServices/TreeSemAgent/tests/test_routing_benchmark.py` — benchmark calculation and report-safety tests.
- `deploy/docker/routing-export.Dockerfile` — reproducible heavy offline export environment, never used by the Agent runtime.

**Modify:**

- `PythonServices/TreeSemAgent/agent/embedding_provider.py` — retain the heavy provider as offline golden reference and add shared text-prefix/hash helpers only.
- `PythonServices/TreeSemAgent/agent/routing_config.py` — select ONNX artifacts in semantic modes and preserve optional/required behavior.
- `PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py` — accept an injected/configured embedding backend without duplicating router policy.
- `PythonServices/TreeSemAgent/requirements-routing.txt` — production ONNX-only dependencies.
- `PythonServices/TreeSemAgent/tests/test_server_routing.py` — new settings and provider-factory contract.
- `PythonServices/TreeSemAgent/tests/test_routing_packaging.py` — prove PyTorch is absent from the runtime image and artifacts are read-only.
- `deploy/docker/agent.Dockerfile` — install only routing runtime requirements when semantic routing is enabled.
- `docker-compose.yml` — mount the selected routing artifact read-only and pass backend configuration.
- `.env.example` — document ONNX backend and artifact directory.
- `scripts/prepare-routing-model.sh` — explicit offline export entry point instead of runtime-cache preparation.
- `scripts/prepare_demo.py` — validate the configured routing artifact when a semantic mode is requested.
- `Makefile` — add export, parity, and benchmark entry points.
- `docs/reports/agent-routing-evaluation.md` — record measured ONNX parity and resource results.
- `README.md` — document the lightweight optional semantic runtime.

---

### Task 1: Strict Routing Artifact Loader

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/routing_artifact.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_artifact.py`

**Interfaces:**

- Consumes: a local artifact directory, canonical Task Registry bytes, and an expected backend.
- Produces:

```python
class RoutingArtifactError(RuntimeError): ...

@dataclass(frozen=True)
class RoutingArtifactManifest:
    schema_version: int
    artifact_version: str
    source_model_id: str
    source_model_revision: str
    embedding_backend: str
    onnx_opset: int
    onnx_input_names: tuple[str, ...]
    onnx_output_name: str
    embedding_dimension: int
    intent_example_count: int
    intent_examples_sha256: str
    query_prefix: str
    passage_prefix: str
    max_sequence_length: int
    pooling: str
    normalization: str
    pad_token: str
    pad_token_id: int
    tokenizer_library_version: str
    exporter_version: str
    task_registry_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    intent_embeddings_sha256: str
    golden_routes_sha256: str

@dataclass(frozen=True)
class RoutingArtifact:
    directory: Path
    manifest: RoutingArtifactManifest
    model_path: Path
    tokenizer_path: Path
    intent_embeddings: tuple[tuple[float, ...], ...]

def canonical_intent_examples(registry: TaskRegistry) -> tuple[str, ...]: ...
def intent_examples_sha256(examples: Sequence[str]) -> str: ...
def load_routing_artifact(
    directory: Path,
    *,
    task_registry_path: Path,
    registry: TaskRegistry,
    expected_backend: str,
    expected_model_id: str,
    expected_revision: str,
) -> RoutingArtifact: ...
```

- [ ] **Step 1: Write failing manifest and matrix tests**

Create fixtures in a temporary directory and assert exact acceptance/rejection:

```python
def test_loads_backend_aligned_artifact(self):
    artifact = write_valid_artifact(self.directory, backend="onnx_fp32")
    loaded = load_routing_artifact(
        artifact,
        task_registry_path=self.tasks,
        registry=self.registry,
        expected_backend="onnx_fp32",
        expected_model_id=MODEL,
        expected_revision=REVISION,
    )
    self.assertEqual(loaded.manifest.intent_example_count, len(
        canonical_intent_examples(self.registry)))
    self.assertEqual(len(loaded.intent_embeddings[0]), 384)

def test_rejects_backend_mismatch(self):
    artifact = write_valid_artifact(self.directory, backend="onnx_fp32")
    with self.assertRaisesRegex(RoutingArtifactError, "backend"):
        load_routing_artifact(
            artifact, task_registry_path=self.tasks,
            registry=self.registry, expected_backend="onnx_int8",
            expected_model_id=MODEL, expected_revision=REVISION)

def test_rejects_truncated_float_matrix(self):
    artifact = write_valid_artifact(self.directory, backend="onnx_fp32")
    (artifact / "intent_embeddings.f32").write_bytes(b"\x00" * 7)
    refresh_checksum(artifact, "intent_embeddings.f32")
    with self.assertRaisesRegex(RoutingArtifactError, "matrix size"):
        self.load(artifact)
```

Also cover unknown/missing manifest fields, Schema version, model/revision,
Task Registry SHA, intent-example SHA/order, every subfile checksum, NaN/Inf,
zero-norm rows, incorrect dimensions, path traversal in fixed filenames, and
unsupported ONNX input/output declarations.

- [ ] **Step 2: Run the focused test and confirm the red state**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_artifact -v
```

Expected: FAIL because `agent.routing_artifact` does not exist.

- [ ] **Step 3: Implement strict parsing, checksum validation, and raw f32 loading**

Use exact-field validation and little-endian float32 data:

```python
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _load_matrix(path: Path, rows: int, columns: int):
    expected_bytes = rows * columns * 4
    if path.stat().st_size != expected_bytes:
        raise RoutingArtifactError("intent embedding matrix size mismatch")
    values = numpy.fromfile(path, dtype="<f4").reshape(rows, columns)
    if not numpy.isfinite(values).all():
        raise RoutingArtifactError("intent embeddings must be finite")
    norms = numpy.linalg.norm(values, axis=1)
    if numpy.any(norms <= 0) or not numpy.isfinite(norms).all():
        raise RoutingArtifactError("intent embeddings must have positive norm")
    return tuple(tuple(float(value) for value in row) for row in values)
```

Compute the canonical example sequence using the same sorted-scope iteration
already used by `SemanticScorer`. Never trust a manifest-supplied path; all
artifact child names are fixed constants joined to the validated directory.

- [ ] **Step 4: Run artifact and existing routing tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_artifact -v
make routing-unit
```

Expected: artifact tests PASS and the existing routing suite remains green.

- [ ] **Step 5: Commit the loader**

```bash
git add PythonServices/TreeSemAgent/agent/routing_artifact.py \
        PythonServices/TreeSemAgent/tests/test_routing_artifact.py
git commit -m "feat: validate versioned agent routing artifacts"
```

---

### Task 2: Deterministic FP32 Export and Backend-Aligned Intent Embeddings

**Files:**

- Create: `PythonServices/TreeSemAgent/tools/__init__.py`
- Create: `PythonServices/TreeSemAgent/tools/export_routing_artifact.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_export.py`
- Create: `PythonServices/TreeSemAgent/requirements-routing-export.txt`
- Create: `deploy/docker/routing-export.Dockerfile`
- Modify: `PythonServices/TreeSemAgent/agent/embedding_provider.py`

**Interfaces:**

- Consumes: pinned source model/revision, Task Registry, frozen parity corpus,
  thresholds, backend (`onnx_fp32` or `onnx_int8`), and output directory.
- Produces:

```python
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

def export_routing_artifact(settings: ExportSettings) -> Path: ...
def quantize_dynamic_int8(source: Path, destination: Path) -> None: ...
```

- [ ] **Step 1: Add deterministic helper tests before importing heavy frameworks**

Test the pure portions with fake encoders and file bytes:

```python
def test_same_inputs_produce_same_artifact_version(self):
    first = build_manifest_payload(self.inputs)
    second = build_manifest_payload(self.inputs)
    self.assertEqual(artifact_version(first), artifact_version(second))

def test_backend_generates_its_own_intent_embeddings(self):
    encoder = RecordingEncoder([[1.0, 0.0], [0.0, 1.0]])
    write_intent_embeddings(self.registry, encoder, self.output)
    self.assertEqual(encoder.calls, [[
        "passage: " + text for text in canonical_intent_examples(self.registry)
    ]])

def test_export_rejects_non_mean_pooling_source(self):
    with self.assertRaisesRegex(ExportError, "mean pooling"):
        validate_sentence_transformer_modules(FakeModel(pooling="max"))
```

Cover module chain, embedding dimension 384, max length 512, pad ID 1,
right-padding/truncation, model revision, opset 17, stable JSON rendering,
manifest checksum creation, and overwrite refusal for a non-empty output.

- [ ] **Step 2: Run the export tests and confirm they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_export -v
```

Expected: FAIL because the export module does not exist.

- [ ] **Step 3: Implement the exact E5 wrapper and FP32 exporter**

Validate the real Sentence Transformers module chain (`Transformer`, mean
`Pooling(include_prompt=True)`, `Normalize`) before exporting. The graph body
must use the real underlying AutoModel and attention-mask mean pooling:

```python
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
```

Export with names and dynamic axes fixed in the manifest:

```python
torch.onnx.export(
    wrapper,
    (encoded["input_ids"], encoded["attention_mask"]),
    model_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["sentence_embedding"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "sentence_embedding": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

Run `onnx.checker.check_model` and shape inference. Load the exported graph in
ONNX Runtime, encode the canonical `passage: ` examples with that graph, and
write those selected-backend vectors as little-endian `intent_embeddings.f32`.
Generate `golden_routes.json` from the frozen 150 cases and the existing
PyTorch provider; do not change their labels or messages.

- [ ] **Step 4: Add explicit INT8 export without changing the FP32 path**

Use ONNX Runtime dynamic quantization:

```python
quantize_dynamic(
    model_input=str(fp32_path),
    model_output=str(int8_path),
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)
```

After quantization, validate the graph and generate a fresh intent matrix by
running the INT8 graph. Never copy the FP32 intent matrix into the INT8
artifact.

- [ ] **Step 5: Split runtime and exporter requirements**

Set `requirements-routing-export.txt` to:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
sentence-transformers==5.7.0
transformers==4.57.6
onnx==1.17.0
onnxruntime==1.20.1
numpy==2.0.1
tokenizers==0.22.2
```

Do not change the production runtime requirements in this task.

- [ ] **Step 6: Add the reproducible offline exporter image**

Create a dedicated image that is never inherited by the production Agent:

```dockerfile
FROM python:3.10-slim-bookworm
WORKDIR /app
COPY PythonServices/TreeSemAgent/requirements.txt /tmp/requirements.txt
COPY PythonServices/TreeSemAgent/requirements-routing-export.txt /tmp/requirements-routing-export.txt
RUN python -m pip install --no-cache-dir \
      -r /tmp/requirements.txt -r /tmp/requirements-routing-export.txt
COPY PythonServices/TreeSemAgent /app
ENTRYPOINT ["python", "-m", "tools.export_routing_artifact"]
```

The image is intentionally heavy but only exists for explicit offline artifact
preparation. It must not be referenced by the production `agent` service.

- [ ] **Step 7: Run pure tests and one real local FP32 export smoke test**

Run the unit test, then run the exporter in the already prepared local model
cache. The output stays under ignored `artifacts/agent-routing/`:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_export -v

docker build -f deploy/docker/routing-export.Dockerfile \
  -t treesem-routing-export:test .

install -d artifacts/agent-routing
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$HOME/.cache/huggingface:/models/huggingface:ro" \
  -v "$PWD/artifacts/agent-routing:/output" \
  treesem-routing-export:test \
  --backend onnx_fp32 \
  --model intfloat/multilingual-e5-small \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --tasks /app/config/tasks.yaml \
  --parity-cases /app/evaluation/routing_cases.json \
  --thresholds /app/config/routing_thresholds.json \
  --output-root /output
```

Expected: `onnx.checker` passes, the manifest and five business files exist,
and rerunning to a clean temporary root produces the same artifact version and
business-file checksums.

- [ ] **Step 8: Commit exporter and dependency split**

```bash
git add PythonServices/TreeSemAgent/tools \
        PythonServices/TreeSemAgent/tests/test_routing_export.py \
        PythonServices/TreeSemAgent/requirements-routing-export.txt \
        PythonServices/TreeSemAgent/agent/embedding_provider.py \
        deploy/docker/routing-export.Dockerfile
git commit -m "feat: export deterministic ONNX routing artifacts"
```

---

### Task 3: ONNX Runtime Embedding Provider

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/onnx_embedding_provider.py`
- Create: `PythonServices/TreeSemAgent/tests/test_onnx_embedding_provider.py`
- Modify: `PythonServices/TreeSemAgent/requirements-routing.txt`

**Interfaces:**

- Consumes: a validated `RoutingArtifact` and injectable tokenizer/session
  factories for tests.
- Produces:

```python
class OnnxEmbeddingProvider:
    def __init__(
        self,
        artifact: RoutingArtifact,
        *,
        tokenizer_factory: Callable[[Path], Any] | None = None,
        session_factory: Callable[[Path], Any] | None = None,
    ): ...

    def encode_examples(self, texts: Sequence[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...
```

- [ ] **Step 1: Write failing provider tests with fake tokenizer and session**

Required cases include:

```python
def test_query_uses_prefix_and_int64_onnx_inputs(self):
    provider = self.provider(output=[[3.0, 4.0]])
    vector = provider.encode_query("解释结果")
    self.assertEqual(self.tokenizer.texts, ["query: 解释结果"])
    self.assertEqual(self.session.feed_names, {
        "input_ids", "attention_mask"})
    self.assertEqual(self.session.feed_dtypes, {numpy.dtype("int64")})
    self.assertAlmostEqual(vector[0], 0.6)
    self.assertAlmostEqual(vector[1], 0.8)

def test_examples_come_from_backend_aligned_artifact(self):
    provider = self.provider(intent_embeddings=((1.0, 0.0),))
    self.assertEqual(provider.encode_examples(self.examples), [[1.0, 0.0]])
    self.assertEqual(self.session.calls, 0)

def test_rejects_unexpected_example_sequence(self):
    provider = self.provider()
    with self.assertRaisesRegex(ValueError, "intent examples"):
        provider.encode_examples(["changed example"])
```

Also reject empty input, bool/non-string text, output rank/shape mismatch,
NaN/Inf, zero-norm output, wrong session I/O names, and tokenizer sequences
longer than 512 after truncation.

- [ ] **Step 2: Run the provider tests and confirm the red state**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_onnx_embedding_provider -v
```

Expected: FAIL because the provider module does not exist.

- [ ] **Step 3: Implement tokenizer and ORT Session initialization**

Load `tokenizer.json` locally, enable the manifest's truncation and right
padding, and build one CPU session:

```python
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session = onnxruntime.InferenceSession(
    str(artifact.model_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
```

Use `Tokenizer.encode_batch`, construct rectangular `numpy.int64` input IDs
and masks, call `session.run([output_name], feeds)`, validate the shape, and
normalize defensively before returning Python floats. Set
`TOKENIZERS_PARALLELISM=false` before tokenizer work unless the operator has
already set a value.

- [ ] **Step 4: Replace production routing requirements**

Set `requirements-routing.txt` to exactly:

```text
onnxruntime==1.20.1
numpy==2.0.1
tokenizers==0.22.2
```

PyTorch, Transformers, Sentence Transformers, and the PyTorch extra index must
not appear in this runtime file.

- [ ] **Step 5: Run focused and semantic-scoring tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_onnx_embedding_provider \
  tests.test_semantic_routing -v
```

Expected: all tests PASS with no routing-policy changes.

- [ ] **Step 6: Commit the runtime provider**

```bash
git add PythonServices/TreeSemAgent/agent/onnx_embedding_provider.py \
        PythonServices/TreeSemAgent/tests/test_onnx_embedding_provider.py \
        PythonServices/TreeSemAgent/requirements-routing.txt
git commit -m "feat: run agent semantic embeddings with ONNX"
```

---

### Task 4: Runtime Configuration and Safe Degradation

**Files:**

- Modify: `PythonServices/TreeSemAgent/agent/routing_config.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_server_routing.py`
- Modify: `.env.example`

**Interfaces:**

- Consumes: the Task Registry and the Task 1/3 loader/provider.
- Produces:

```python
class RoutingEmbeddingBackend(str, Enum):
    ONNX_FP32 = "onnx_fp32"
    ONNX_INT8 = "onnx_int8"

@dataclass(frozen=True)
class RoutingSettings:
    # existing fields remain
    embedding_backend: RoutingEmbeddingBackend
    artifact_dir: Path

def create_onnx_provider(
    settings: RoutingSettings,
    registry: TaskRegistry,
) -> OnnxEmbeddingProvider: ...
```

- [ ] **Step 1: Update tests first for rule, optional, and required modes**

Add these expectations:

```python
def test_rule_mode_does_not_require_artifact(self):
    settings = self.settings(RoutingMode.RULE, artifact_dir=Path("missing"))
    runtime = build_routing_runtime(
        settings,
        provider_factory=lambda *_: self.fail("provider constructed"))
    self.assertFalse(runtime.semantic_available)

def test_optional_mode_degrades_on_artifact_error(self):
    runtime = build_routing_runtime(
        self.settings(RoutingMode.HYBRID_OPTIONAL),
        provider_factory=lambda *_: (_ for _ in ()).throw(
            RoutingArtifactError("checksum mismatch")))
    self.assertEqual(runtime.degradation_reason,
                     "semantic_initialization_failed")

def test_required_mode_propagates_artifact_error(self):
    with self.assertRaises(RoutingArtifactError):
        build_routing_runtime(
            self.settings(RoutingMode.HYBRID_REQUIRED),
            provider_factory=self.failing_provider)
```

Test invalid backend, missing artifact configuration in semantic modes,
environment path parsing, model/revision mismatch, and health/readiness
metadata. Preserve injected-loop tests.

- [ ] **Step 2: Run tests and confirm constructor/signature failures**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_server_routing -v
```

Expected: FAIL because the settings do not yet contain backend/artifact data.

- [ ] **Step 3: Implement backend selection without changing mode semantics**

Change the provider factory boundary to:

```python
ProviderFactory = Callable[[RoutingSettings, TaskRegistry], EmbeddingProvider]

def create_onnx_provider(settings, registry):
    artifact = load_routing_artifact(
        settings.artifact_dir,
        task_registry_path=settings.task_registry,
        registry=registry,
        expected_backend=settings.embedding_backend.value,
        expected_model_id=settings.model,
        expected_revision=settings.revision,
    )
    return OnnxEmbeddingProvider(artifact)
```

`build_routing_runtime` must return before constructing any provider in `rule`
mode. Its existing optional `try/except` remains the only startup-degradation
boundary; required mode continues to re-raise.

- [ ] **Step 4: Document environment settings**

Add:

```text
TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND=onnx_int8
TREESEM_AGENT_ROUTING_ARTIFACT_DIR=/absolute/path/to/routing/artifact
```

Keep the default mode `rule`. Explain that the Artifact directory is required
only when a hybrid mode is selected.

- [ ] **Step 5: Run configuration and full routing tests**

Run:

```bash
make routing-unit
```

Expected: all routing tests PASS.

- [ ] **Step 6: Commit configuration integration**

```bash
git add PythonServices/TreeSemAgent/agent/routing_config.py \
        PythonServices/TreeSemAgent/tests/test_server_routing.py \
        .env.example
git commit -m "feat: configure ONNX agent routing backends"
```

---

### Task 5: Frozen-Set Parity Runner

**Files:**

- Create: `PythonServices/TreeSemAgent/evaluation/compare_routing_backends.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py`
- Modify: `PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: frozen cases, Task Registry, thresholds, PyTorch golden provider,
  and an ONNX artifact provider.
- Produces a JSON report containing:

```text
schema_version
case_count
embedding_backend
source_model_id/revision
artifact_version
maximum_embedding_absolute_delta
minimum_embedding_cosine_similarity
maximum_top_similarity_delta
maximum_margin_delta
route_match_count
route_mismatch_count
route_mismatches
quality_metrics
passed
```

- [ ] **Step 1: Write failing comparison tests with deterministic fake scorers**

```python
def test_report_passes_identical_routes(self):
    report = compare_backends(self.cases, self.golden, self.candidate)
    self.assertEqual(report["route_mismatch_count"], 0)
    self.assertTrue(report["passed"])

def test_any_route_change_fails_parity(self):
    candidate = FakeBackend(routes={"route-001": "history"})
    report = compare_backends(self.cases, self.golden, candidate)
    self.assertEqual(report["route_mismatch_count"], 1)
    self.assertFalse(report["passed"])
```

Also test finite deltas, deterministic mismatch ordering, all 150 cases being
required, safety/unknown/compositional 100% gates, and non-zero exit status on
failure.

- [ ] **Step 2: Run the parity tests and confirm the red state**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_routing_backend_parity -v
```

Expected: FAIL because the comparison module does not exist.

- [ ] **Step 3: Implement the runner using shared routing policy**

Do not copy threshold policy into a third implementation. Construct two
`SemanticScorer` instances and use the same `_hybrid_scope`/shared helper for
Safety Gate, Rule Router, and threshold application. Compare embeddings for
all canonical intent examples and all case queries, then compare scores and
final scopes.

Add provider selection to `run_routing_evaluation.py`:

```text
--embedding-backend sentence_transformers|onnx_fp32|onnx_int8
--artifact-dir PATH
```

The default remains the existing golden provider for calibration tooling;
production code does not import it.

- [ ] **Step 4: Add Make entry points**

Add:

```make
routing-export-fp32:
	... python3 -m tools.export_routing_artifact --backend onnx_fp32 ...

routing-export-int8:
	... python3 -m tools.export_routing_artifact --backend onnx_int8 ...

routing-parity:
	... python3 -m evaluation.compare_routing_backends ...
```

All paths and the pinned model revision come from existing Make variables; the
artifact directory is overridable with `ROUTING_ARTIFACT_DIR`.

- [ ] **Step 5: Run tests and the real FP32 150-case parity gate**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_routing_backend_parity \
  tests.test_routing_evaluation -v

make routing-parity ROUTING_ARTIFACT_DIR=<generated-fp32-artifact>
```

Expected: 150 cases evaluated, zero final-route mismatches, all hard gates
pass, and the report is written under ignored
`artifacts/evaluation/routing/`.

- [ ] **Step 6: Commit parity tooling**

```bash
git add PythonServices/TreeSemAgent/evaluation/compare_routing_backends.py \
        PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
        PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py \
        Makefile
git commit -m "test: verify ONNX agent routing parity"
```

---

### Task 6: Production Packaging and Offline Artifact Mount

**Files:**

- Modify: `deploy/docker/agent.Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `scripts/prepare-routing-model.sh`
- Modify: `scripts/prepare_demo.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_routing_packaging.py`

**Interfaces:**

- Consumes: Task 2 exporter and a selected artifact directory.
- Produces: an ONNX-only semantic Agent image and a Compose read-only mount.

- [ ] **Step 1: Rewrite packaging expectations before Docker changes**

Tests must assert:

```python
def test_runtime_requirements_exclude_training_frameworks(self):
    requirements = self.runtime_requirements()
    self.assertIn("onnxruntime==1.20.1", requirements)
    self.assertIn("tokenizers==0.22.2", requirements)
    self.assertNotIn("torch", requirements)
    self.assertNotIn("sentence-transformers", requirements)
    self.assertNotIn("transformers==", requirements)

def test_compose_mounts_routing_artifact_read_only(self):
    compose = self.compose_text()
    self.assertIn("TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND", compose)
    self.assertIn("TREESEM_AGENT_ROUTING_ARTIFACT_DIR: /routing/artifact", compose)
    self.assertIn("/routing/artifact:ro", compose)
```

Also assert no `snapshot_download` in the Dockerfile, explicit export-only
requirements, rule-mode defaults, and `prepare_demo.py` failure when a hybrid
mode points to an invalid artifact.

- [ ] **Step 2: Run packaging tests and confirm they fail against PyTorch**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_packaging -v
```

Expected: FAIL because the current runtime requirements contain PyTorch and
Compose mounts the Hugging Face cache.

- [ ] **Step 3: Change Docker and Compose to ONNX-only runtime packaging**

Keep `TREESEM_INSTALL_SEMANTIC_ROUTING` as the optional build switch, but have
it install the new three-package runtime file. Remove the Agent's Hugging Face
cache mount and offline Transformers environment. Add:

```yaml
environment:
  TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND: ${TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND:-onnx_int8}
  TREESEM_AGENT_ROUTING_ARTIFACT_DIR: /routing/artifact
  TOKENIZERS_PARALLELISM: "false"
volumes:
  - ${TREESEM_AGENT_ROUTING_ARTIFACT_DIR:-./artifacts/agent-routing/disabled}:/routing/artifact:ro
```

Rule mode must not inspect the empty default mount.

- [ ] **Step 4: Make preparation explicit and fail-fast**

Change `prepare-routing-model.sh` into a thin invocation of the Task 2 exporter
image with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` by default. The script
builds `deploy/docker/routing-export.Dockerfile`, mounts the pinned local Hugging
Face cache read-only, mounts the output root read-write under the current user,
and accepts:

```text
TREESEM_ROUTING_EXPORT_BACKEND=onnx_fp32|onnx_int8
TREESEM_ROUTING_OUTPUT_ROOT=artifacts/agent-routing
```

It must print the resulting artifact directory and never run during Agent
startup. `prepare_demo.py` validates `manifest.json` and checksums when the
configured mode is hybrid; rule mode creates/accepts the ignored disabled
mount without requiring a model.

- [ ] **Step 5: Run packaging, Compose, and image-content checks**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_packaging -v

docker compose config --quiet

docker build \
  --build-arg TREESEM_INSTALL_SEMANTIC_ROUTING=true \
  -f deploy/docker/agent.Dockerfile \
  -t treesem-agent-routing-onnx:test .

docker run --rm --entrypoint python treesem-agent-routing-onnx:test -c \
  'import onnxruntime, tokenizers, numpy; import importlib.util; assert importlib.util.find_spec("torch") is None; assert importlib.util.find_spec("sentence_transformers") is None'
```

Expected: all commands exit 0 and neither heavy runtime is installed.

- [ ] **Step 6: Commit packaging**

```bash
git add deploy/docker/agent.Dockerfile docker-compose.yml \
        scripts/prepare-routing-model.sh scripts/prepare_demo.py \
        PythonServices/TreeSemAgent/tests/test_routing_packaging.py
git commit -m "build: package lightweight ONNX agent routing"
```

---

### Task 7: Runtime Benchmark and Reproducible Resource Report

**Files:**

- Create: `PythonServices/TreeSemAgent/evaluation/benchmark_routing_runtime.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_benchmark.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: an artifact, Task Registry, warm-up count, measurement count, and
  fixed benchmark queries.
- Produces JSON with:

```text
artifact_version/backend
artifact_size_bytes
cold_initialization_ms
first_route_ms
warmed_p50/p95/p99_route_ms
maximum_rss_kib
case_count/warmup_count
host/platform/python/onnxruntime metadata
```

- [ ] **Step 1: Write failing percentile, footprint, and report tests**

```python
def test_total_footprint_includes_image_and_artifact(self):
    report = deployment_footprint(image_bytes=300, artifact_bytes=130)
    self.assertEqual(report["total_deployment_bytes"], 430)

def test_percentiles_use_observed_samples(self):
    self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 4.0)
```

Test stable JSON fields, zero/negative count validation, recursive artifact
size, Linux `ru_maxrss` units, and that raw queries are absent from the report.

- [ ] **Step 2: Run benchmark tests and confirm the red state**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest tests.test_routing_benchmark -v
```

Expected: FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement process and deployment measurements**

Time provider/artifact construction separately from first and warmed routes.
Use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` on Linux and
`time.perf_counter_ns`. Read Docker image bytes from an explicit CLI argument
or a caller-provided `docker image inspect --format '{{.Size}}'` value; do not
silently treat a missing image as zero.

The benchmark output must calculate both:

```python
total_deployment_bytes = container_image_bytes + artifact_size_bytes
relative_change = (candidate - baseline) / baseline
```

- [ ] **Step 4: Add and run the benchmark target**

Add:

```make
routing-benchmark:
	PYTHONPATH=$(ROUTING_ROOT) python3 -m evaluation.benchmark_routing_runtime \
	  --artifact-dir $(ROUTING_ARTIFACT_DIR) \
	  --tasks $(ROUTING_TASKS) --warmup 100 --iterations 1000 \
	  --output artifacts/evaluation/routing/onnx-runtime-benchmark.json
```

Run it for FP32 and later INT8 on the same host. Record 100 warm-ups and 1000
measured routes.

- [ ] **Step 5: Commit benchmark tooling**

```bash
git add PythonServices/TreeSemAgent/evaluation/benchmark_routing_runtime.py \
        PythonServices/TreeSemAgent/tests/test_routing_benchmark.py Makefile
git commit -m "test: benchmark ONNX agent routing resources"
```

---

### Task 8: Full FP32/INT8 Validation and Pre-Corpus Checkpoint

**Files:**

- Modify: `docs/reports/agent-routing-evaluation.md`
- Modify: `README.md`
- Modify: `docs/treesem-interview-guide.md`

**Interfaces:**

- Consumes: both generated artifacts, parity reports, image measurements, and
  runtime benchmarks.
- Produces: selected pre-quality candidate, evidence report, and the mandatory
  handoff to the user for corpus generation.

- [ ] **Step 1: Export both backends and run frozen-set parity**

Run:

```bash
make routing-export-fp32
make routing-parity ROUTING_ARTIFACT_DIR=<fp32-artifact>
make routing-export-int8
make routing-parity ROUTING_ARTIFACT_DIR=<int8-artifact>
```

Expected: FP32 has zero route mismatches. INT8 must also have zero frozen-set
route mismatches to remain a deployment candidate; otherwise mark INT8
rejected and keep FP32.

- [ ] **Step 2: Run full deterministic and concurrency regression**

Run:

```bash
make routing-unit
make routing-load-smoke
make verify
```

Expected: all routing tests pass, executor smoke passes 20 rounds, all C++
tests pass, and all 64 deterministic Agent cases pass.

- [ ] **Step 3: Build images and collect comparable resource evidence**

Build the base, old PyTorch semantic, and new ONNX semantic images from the
same commit/environment where possible. Record:

```text
image bytes
artifact bytes
image + artifact bytes
steady/maximum RSS
cold initialization
first route
warmed p50/p95/p99
```

Do not copy earlier measurements into the new result table as if they were
produced by the current build. Label any retained historical baseline clearly.

- [ ] **Step 4: Update documentation with facts, not targets**

Document:

- exact backend selected at this checkpoint;
- FP32 and INT8 parity outcome;
- absolute and relative resource changes;
- tokenizer and backend-aligned intent-embedding design;
- why `rule` remains the default until Quality Set evidence exists;
- FP32 rollback when INT8 fails;
- no paid real-LLM rerun when route decisions are identical.

The interview guide should frame this as a measured deployment optimization,
not claim that moving files to a volume alone reduced total footprint.

- [ ] **Step 5: Run final static and repository checks**

Run:

```bash
git diff --check
git status --short
if git ls-files | rg '\.(onnx|pt|pth|safetensors|f32)$'; then exit 1; fi
docker compose config --quiet
```

Expected: no whitespace errors, no generated model files tracked, and Compose
configuration valid.

- [ ] **Step 6: Commit the pre-corpus checkpoint**

```bash
git add README.md docs/reports/agent-routing-evaluation.md \
        docs/treesem-interview-guide.md
git commit -m "docs: report ONNX agent routing optimization"
```

- [ ] **Step 7: Mandatory stop and user handoff**

Stop execution. Report to the user:

```text
FP32 parity result
INT8 parity/result and selected candidate
image/artifact/total-footprint/RSS/cold-start/latency measurements
remaining risks
Quality Set Schema
target count by category and split
generation prompt
deduplication and review checklist
estimated generation token budget
```

Do not create, generate, or label any new Quality Set cases. Wait for the user
to arrange generation and provide the candidate corpus or give a new explicit
authorization.

---

## Deferred Task: User-Managed Routing Quality Set

This section is intentionally not executable in the current run. After the
user supplies generated candidates, create a separate reviewed plan for:

1. Schema validation and normalization.
2. Exact and semantic deduplication against the frozen 150 cases.
3. Human review of multi-intent, OOD, adversarial, and safety cases.
4. Fixed calibration/held-out split before threshold work.
5. FP32 and INT8 quality evaluation.
6. Final backend promotion decision.
7. Targeted real-LLM reruns only for changed/adjacent routing cases.

No task in this implementation plan authorizes LLM corpus generation.
