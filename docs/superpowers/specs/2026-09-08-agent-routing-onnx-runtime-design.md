# treeSem Agent Routing ONNX Runtime Design

## Status

Approved in chat on 2026-09-08. This design lightens the existing optional
semantic router without changing its routing policy, Workflow definitions,
authorization model, Tool implementations, or medical response policy.

Implementation must stop before generating the new Routing Quality Set. The
user will arrange corpus generation and provide the generated candidates or
explicitly authorize a later continuation. No implementation task may call an
LLM to generate that corpus automatically.

## Objective

Replace the deployment-time Sentence Transformers and PyTorch embedding
runtime with a versioned, locally loaded ONNX embedding provider:

```text
Safety Gate
  -> high-precision Rule Router
  -> bounded Semantic Routing Executor
  -> ONNX embedding provider
  -> existing similarity and ambiguity gates
  -> deterministic Workflow or guarded UNKNOWN path
```

The change is successful when it preserves existing routing behavior and
safety boundaries while materially reducing runtime memory, cold-start time,
container image size, and total deployment footprint.

## Current Baseline

The current semantic candidate uses:

```text
sentence-transformers==5.7.0
transformers==4.57.6
torch==2.5.1+cpu
intfloat/multilingual-e5-small
revision 614241f622f53c4eeff9890bdc4f31cfecc418b3
```

Measured on the current development host:

```text
base Agent image                         about 166.7 MiB
semantic Agent image                     about 1244.8 MiB
semantic image increase                  about 1078.1 MiB
semantic runtime RSS increase            about 745.6 MiB
semantic cold initialization             about 6.33 seconds
held-out known-intent accuracy            60.0%
held-out macro F1                        72.685%
safety/unknown/compositional pass rate   100%
```

The semantic router remains optional because its current routing-quality gain
is modest. This optimization does not by itself justify promoting
`hybrid_optional` over the default `rule` mode.

## Design Principles

1. Change the embedding runtime, not the routing strategy.
2. Safety and high-precision rules always run before semantic inference.
3. Blocking ONNX inference stays behind the existing bounded routing
   executor; FastAPI's EventLoop never runs it directly.
4. The deployed query embeddings and intent-example embeddings must come from
   the same model revision, tokenizer configuration, ONNX graph, precision,
   and backend.
5. PyTorch remains an offline golden reference and export dependency, not a
   production Agent dependency.
6. ONNX FP32 must prove parity before INT8 is considered.
7. INT8 is a candidate optimization. If it fails quality or parity gates, the
   deployable result is FP32 ONNX.
8. Models and tokenizers are immutable local artifacts with checksums. Runtime
   startup never downloads or silently replaces them.
9. `hybrid_optional` fails soft; `hybrid_required` fails fast.
10. Report container image, routing artifact, total deployment footprint, RSS,
    cold start, and request latency separately.
11. A generated routing corpus is candidate data, not trusted ground truth.
12. Avoid a routing microservice until measured multi-replica duplication
    justifies its network and operational cost.

## Request Flow

The existing control flow remains unchanged:

```text
Agent request
  |
  v
deterministic safety policy
  |-- refusal ------------------------> no LLM or semantic routing
  `-- allowed
        |
        v
high-precision business/Skill rules
  |-- hit ----------------------------> deterministic Workflow/Skill path
  `-- miss
        |
        v
bounded routing executor
        |
        v
ONNX embedding provider
  -> tokenize "query: <message>"
  -> ONNX Session.Run
  -> normalized sentence embedding
        |
        v
existing similarity, margin, and multi-intent gates
  |-- gates pass ---------------------> deterministic Workflow
  `-- gates fail/error/timeout -------> UNKNOWN guarded Agent path
```

The routing executor remains a fixed-size `ThreadPoolExecutor` combined with
bounded admission and `asyncio.wait_for`. No new general scheduler subsystem is
introduced.

## Exported Embedding Behavior

The export must reproduce the behavior actually exercised by
`SentenceTransformer.encode`, rather than implementing an assumed E5 formula.
The following are frozen and recorded in the artifact manifest:

- tokenizer files and tokenizer library version;
- `query: ` prefix for user queries;
- `passage: ` prefix for intent examples;
- special-token behavior;
- padding and attention masks;
- maximum sequence length and truncation policy;
- Transformer output selection;
- attention-mask-aware pooling;
- L2 normalization;
- embedding dimension and ONNX input/output names.

The preferred ONNX graph accepts `input_ids` and `attention_mask` and emits the
final normalized sentence embedding. If the source model requires an
additional tokenizer output such as `token_type_ids`, the manifest and runtime
must expose it explicitly; the runtime must not fabricate unused inputs.

The export is accepted only after golden tests demonstrate equivalence with
the pinned Sentence Transformers behavior for English, Chinese, mixed-language,
long/truncated, punctuation-heavy, and special-token inputs.

## Routing Artifact

Each deployable backend has a separate immutable artifact directory:

```text
artifacts/agent-routing/<artifact-version>/
├── manifest.json
├── model.onnx
├── tokenizer.json
├── tokenizer_config.json
├── intent_embeddings.f32
└── golden_routes.json
```

The artifact version is deterministic over the model, tokenizer, backend,
Task Registry, exporter, and selected ONNX graph.

`manifest.json` records at least:

```text
schema_version
artifact_version
source_model_id
source_model_revision
embedding_backend                 # onnx_fp32 or onnx_int8
onnx_opset
onnx_input_names
onnx_output_name
embedding_dimension
query_prefix
passage_prefix
max_sequence_length
pooling
normalization
tokenizer_library_version
exporter_version
task_registry_sha256
model_sha256
tokenizer_sha256
intent_embeddings_sha256
golden_routes_sha256
```

An FP32 artifact contains FP32-generated intent embeddings. An INT8 artifact
contains embeddings generated by that exact INT8 graph. Mixing PyTorch/FP32
intent embeddings with INT8 query embeddings is forbidden.

Intent embeddings are generated offline to avoid embedding all registered
examples during every Agent startup. The runtime validates the Task Registry
checksum and the expected scope/example order before using them.

Generated model artifacts remain excluded from Git and are mounted read-only
at runtime. The manifest Schema, exporter, tests, and synthetic golden inputs
are committed.

## Runtime Provider

The production provider owns:

- a locally loaded tokenizer;
- one immutable ONNX Runtime Session;
- the validated intent-embedding matrix;
- the selected artifact manifest.

It implements the current embedding boundary and returns finite normalized
vectors. It does not own routing thresholds or choose a Workflow.

Runtime defaults:

```text
ONNX execution provider: CPU
ORT intra-op threads:    1
ORT inter-op threads:    1
TOKENIZERS_PARALLELISM:  false
```

The existing outer routing executor controls concurrency. ORT must not create
an additional unbounded layer of worker threads that multiplies the configured
routing workers.

The first implementation may keep a proven tokenizer runtime dependency when
that is necessary for parity. Removing PyTorch, Transformers model execution,
and Sentence Transformers from the production image is the primary target;
removing every tokenizer-related package is secondary.

## Configuration

The existing mode remains authoritative:

```text
TREESEM_AGENT_ROUTING_MODE=rule|hybrid_optional|hybrid_required
```

Semantic modes add:

```text
TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND=onnx_fp32|onnx_int8
TREESEM_AGENT_ROUTING_ARTIFACT_DIR=/artifacts/agent-routing/<version>
```

The legacy Sentence Transformers provider is available only to offline export,
parity, and development tooling. It is not installed in the production Agent
image.

Startup behavior:

```text
rule
  -> never loads an ONNX artifact

hybrid_optional + missing/corrupt/incompatible artifact
  -> start successfully in Rule-only degradation
  -> readiness/health metadata expose semantic unavailability

hybrid_required + missing/corrupt/incompatible artifact
  -> fail startup with a safe configuration error
```

Per-request behavior:

```text
executor overload/timeout/closure or ORT inference failure
  + hybrid_optional -> UNKNOWN for that request
  + hybrid_required -> typed routing failure
```

No error response or log exposes the user message, tokenizer contents, local
absolute paths, or embedding vectors.

## Dependency and Container Boundary

Dependencies are split by purpose:

```text
offline export/golden environment
  torch
  transformers
  sentence-transformers
  onnx
  onnxruntime

production Agent semantic runtime
  onnxruntime
  tokenizer dependency proven by parity tests
  numpy
```

The Agent image and routing artifact are versioned independently. Runtime
Compose mounts the selected artifact read-only. The preparation command
downloads or exports only when explicitly invoked; service startup stays
offline.

Deployment reporting must include:

```text
container image size
routing artifact size
total deployment disk footprint
steady-state and peak RSS
cold initialization time
first-request and warmed request latency
```

Moving the model from the image into a volume is not counted as the whole
optimization. The total footprint and RSS must both be reported.

## FP32 and INT8 Promotion Ladder

Promotion follows this order:

```text
pinned PyTorch golden behavior
  -> export FP32 ONNX
  -> tokenizer/embedding/route parity
  -> package and benchmark FP32 runtime
  -> export dynamic INT8 candidate
  -> regenerate intent embeddings with INT8
  -> parity and routing-quality evaluation
  -> select INT8 only if every hard gate passes
```

FP32 failure stops the implementation from promoting any ONNX backend.

INT8 failure does not invalidate the project. The release uses FP32 ONNX and
records the failed INT8 experiment and evidence.

## Evaluation Data Separation

### Frozen Parity Set

The existing 150 routing cases are frozen as the Parity Set. Its purpose is:

> Did the new runtime change the behavior of the implementation it replaces?

It is not expanded, rewritten, or used to justify the general quality of the
router.

Hard expectations:

- FP32 ONNX produces the same final route for all 150 cases.
- Safety, unknown, and compositional behavior remain 100% correct.
- Score and embedding deltas are recorded within explicit numeric tolerances.
- A deployable INT8 artifact must not introduce a Parity Set route regression;
  otherwise the release uses FP32.

### Routing Quality Set

A separate 300-400 case candidate corpus evaluates whether the router itself
generalizes. It covers:

- paraphrases of known single intents;
- colloquial, elliptical, and noisy requests;
- hard boundary cases between neighboring intents;
- multi-intent/compositional requests;
- out-of-domain requests;
- adversarial and safety-boundary wording.

The generated candidates undergo exact and semantic deduplication, label and
intent-boundary review, and separation from the frozen Parity Set. Calibration
and held-out partitions are fixed before threshold work begins. The final
held-out partition never participates in threshold selection.

### Mandatory User Gate Before Corpus Generation

Implementation stops after the FP32 runtime and INT8 candidate are ready for
the Routing Quality Set. At that point the assistant reports:

- the completed ONNX and parity results;
- the exact candidate-corpus Schema and target counts;
- the generation prompt and review checklist;
- the expected token budget.

The assistant must not generate the corpus, call a model to generate it, or
continue into quality-set calibration until the user provides the corpus or
explicitly authorizes that later action.

## Verification Gates

### Functional and safety gates

- all current routing unit tests pass;
- all 150 frozen Parity Set cases pass the stated parity requirements;
- all 64 deterministic Agent scenarios pass;
- safety, unknown, and compositional routing remain 100%;
- optional/required startup and request degradation behave as specified;
- executor overload, timeout, shutdown, and 20-round concurrency tests pass;
- logs and Metrics contain no messages, embeddings, patient data, or
  high-cardinality identifiers.

### Routing Quality gates

After the user-provided Quality Set is reviewed and frozen:

- known-intent accuracy does not fall below the current 60.0% baseline;
- macro F1 does not fall below the current 72.685% baseline;
- cross-scope false deterministic routing does not increase;
- safety, OOD, and compositional hard gates remain 100%;
- calibration and held-out results are reported separately.

These are no-regression floors, not claims that the current semantic router is
already good enough for default promotion.

### Resource gates

Report actual absolute values and relative changes on the same host. Initial
targets are:

```text
production semantic Agent image       <= 600 MiB and >= 50% reduction
semantic runtime RSS increase         <= 350 MiB and >= 50% reduction
cold semantic initialization          <= 3 seconds and >= 40% reduction
warmed routing p95                     no more than 20% slower than baseline
```

If host-dependent absolute and relative gates disagree, correctness remains a
hard gate and the report explains the environment. Resource results are never
fabricated or copied across machines.

### Paid real-LLM evaluation

An embedding-runtime replacement does not automatically require another full
paid 64-case Agent evaluation:

- when final routing decisions are identical, deterministic regression is
  sufficient;
- when some decisions differ, rerun only affected and adjacent risk cases;
- a full paid evaluation is reserved for a routing-policy, Prompt, Workflow,
  Tool, or Skill behavior change.

## Rollout and Rollback

1. Keep `rule` as the default while the semantic quality advantage remains
   modest.
2. Make ONNX available only in explicit semantic modes.
3. Use `hybrid_required` in validation to expose missing or corrupt artifacts.
4. Use `hybrid_optional` in demonstrations that need graceful degradation.
5. Retain the previous artifact and the `rule` configuration as immediate
   rollback paths.
6. Do not ship PyTorch and ONNX together merely to perform runtime shadow
   comparison; parity is an offline release gate.

## Explicit Non-goals

This design does not add:

- a routing microservice;
- C++ Transformer inference;
- GPU, CUDA, TensorRT, or FP16 routing;
- online model downloads;
- user-message embedding persistence or caching;
- request batching;
- a new thread-pool framework;
- new routing intents, Workflow definitions, or Tools;
- threshold retuning against the frozen Parity Set;
- automatic LLM corpus generation.

A shared routing/embedding service is reconsidered only when multiple Agent
replicas or multiple consumers measurably duplicate enough model memory to
outweigh an extra service and network hop.

## Delivery Sequence

1. Freeze golden Sentence Transformers behavior and resource measurements.
2. Define artifact Schema and deterministic export checks.
3. Export and validate FP32 ONNX.
4. Implement the ONNX provider and configuration without changing router
   policy.
5. Split production and export dependencies and package the read-only artifact.
6. Run FP32 parity, Agent regression, concurrency, and resource benchmarks.
7. Export and evaluate the INT8 candidate against the frozen Parity Set.
8. Stop and request the user-managed Routing Quality Set generation.
9. After the user supplies or explicitly authorizes the corpus, review, split,
   evaluate, and select FP32 or INT8.
10. Update reports, deployment documentation, and interview evidence with
    measured results.

## Definition of Done

- production semantic routing no longer depends on PyTorch or Sentence
  Transformers;
- the deployed model and intent embeddings are backend-aligned and checksum
  verified;
- FP32 ONNX preserves all frozen Parity Set routes;
- INT8 is promoted only if it satisfies parity, safety, quality, and resource
  gates; otherwise FP32 is used;
- FastAPI's EventLoop remains non-blocking and routing resources remain
  bounded;
- `hybrid_optional` and `hybrid_required` preserve their distinct degradation
  semantics;
- image, artifact, total footprint, RSS, cold start, and latency evidence is
  reproducible;
- no corpus is generated automatically, and implementation stops at the
  agreed user gate;
- the default routing mode changes only after separate quality and end-to-end
  evidence supports that decision.
