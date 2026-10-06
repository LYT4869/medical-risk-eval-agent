# treeSem Agent Hybrid Routing Design

## Status

Approved design derived from `AIModelling/updateplan.md` and the 2026-09-07
review. This design changes routing only; it does not change authorization,
Tool implementations, Workflow semantics, or medical response policy.

## Objective

Evolve the current rule-first Agent routing into a measurable three-stage
router:

```text
Deterministic Safety Gate
  -> Explicit Skill / high-precision business Rule Fast Path
  -> Semantic business-intent fallback
  -> UNKNOWN / guarded Open Agent
```

The router selects an execution policy. It never grants permissions, creates
Tool arguments, or executes Tools.

## Current Baseline

The current Agent is not rule-only. `AgentRunGuard.for_request()` combines
deterministic safety checks and keyword business classification. Known scopes
enter deterministic `WorkflowPlan` stages; unmatched messages enter the
guarded `OPEN_AGENT` mode.

The upgrade separates these responsibilities and adds an evaluated semantic
fallback without replacing the existing safe fallback behavior.

## Design Principles

1. Safety, RBAC, Capability JWT validation, Tool Schema validation, and
   side-effect controls remain deterministic and authoritative.
2. Rules optimize precision and latency, not total intent coverage.
3. Semantic routing runs only after the rule fast path misses.
4. Semantic scores are called `similarity_score` and `margin`, never
   probability or confidence.
5. Ambiguous, compositional, out-of-domain, overloaded, timed-out, or disabled
   semantic routing produces `UNKNOWN`; it never guesses a deterministic
   Workflow.
6. `UNKNOWN` keeps the existing restricted read-only Open Agent Tool set.
7. No query, patient identifier, prediction ID, or trace ID becomes a Metrics
   label.
8. The implementation remains small: one fixed-size executor, one admission
   semaphore, one timeout, and one degradation path. There is no general
   Routing Scheduler subsystem.

## Request Flow

```text
AgentRunRequest.message
  |
  v
SafetyGate.evaluate(message)
  |-- security abuse ----------> deterministic refusal
  |-- unsafe medical request --> deterministic refusal
  `-- allowed
        |
        v
RuleRouter.route(message)
  |-- explicit Skill request --> SKILL
  |-- precise business match --> registered business scope
  `-- miss
        |
        v
SemanticRouter.route(message)
  |-- score + ambiguity gates pass --> registered business scope
  `-- otherwise --------------------> UNKNOWN
        |
        v
AgentRunGuard.for_scope(scope)
        |
        v
WorkflowPlanner
  |-- registered scope --> deterministic stages from TaskRegistry
  |-- SKILL -----------> existing trusted Skill planning
  `-- UNKNOWN ---------> guarded OPEN_AGENT
```

## Components

### Routing types

`routing_types.py` owns shared, dependency-neutral types:

```python
class RequestScope(str, Enum): ...
class RoutingSource(str, Enum):
    RULE = "rule"
    SEMANTIC = "semantic"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class RoutingDecision:
    scope: RequestScope
    source: RoutingSource
    similarity_score: float | None = None
    margin: float | None = None
    reason: str | None = None
```

`SECURITY_ABUSE` and `MEDICAL_REFUSAL` remain valid scopes for refusal results,
but `SemanticRouter` is forbidden from producing them. `SKILL` remains an
explicit rule-only route in this phase.

### Task Registry

`TaskRegistry` loads the six static business scopes from a trusted local YAML
file:

- prediction
- summary
- explanation
- history
- comparison
- knowledge

Each definition contains:

```text
scope
mode
rule_terms
intent_examples
workflow_stages
allowed_tools
```

The Registry is not a Workflow DSL. It centralizes static metadata and rejects
duplicate scopes, unknown keys, unknown Tools, invalid modes, empty semantic
examples, safety scopes, and Workflow stages outside `allowed_tools`.

Complex high-precision predicates remain in `RuleRouter` rather than growing a
general boolean rule language in YAML. Examples include:

- summary terms must coexist with stored-result/read context;
- comparison takes precedence over history;
- stored-prediction explanation takes precedence over general knowledge;
- defensive wording such as “不要编造” is not security abuse.

### Safety gate and Run Guard

`run_guard.py` retains:

- security and unsafe-medical detection;
- Tool narrowing;
- successful-Tool suppression;
- no retry after prediction side effects;
- knowledge attempt limits;
- active Skill intersection.

Business intent classification moves out. After routing,
`AgentRunGuard.for_scope()` derives the initial Tool set from `TaskRegistry`.
Special scopes remain fixed:

```text
SKILL  -> activate_skill
UNKNOWN -> existing read-only native Tools
refusal scopes -> no Tools
```

### Rule router

`RuleRouter` performs normalization and explicit high-precision matching. A
business miss returns `None`, allowing the semantic fallback to run. It never
returns arbitrary Tools.

Skill routing remains rule-only and precedes business semantic routing. The
existing `_skill_plan()` remains responsible for selecting one of the trusted
Skills after the request is classified as `SKILL`.

### Semantic router

The evaluated implementation uses `intfloat/multilingual-e5-small`, pinned to
revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. The model is an evaluated
candidate, not a claim of superiority; the hybrid mode is promoted only after
held-out and targeted real-Agent evaluation.

At startup:

```text
Registry intent examples -> embedding -> L2 normalization -> immutable cache
```

Per request:

```text
query embedding
  -> cosine similarity to all examples
  -> per-scope score = maximum example similarity
  -> sort top scopes
  -> apply similarity, margin, and secondary-intent gates
```

The result is `UNKNOWN` when any of these is true:

```text
top1 < min_similarity
top1 - top2 < min_margin
top2 >= secondary_intent_similarity
```

The third condition prevents a compositional request from being forced into a
single Workflow merely because one intent dominates slightly.

Only the six registered business scopes are eligible. Skill and refusal scopes
cannot be semantic outputs.

### Lightweight bounded execution adapter

Embedding inference is blocking and must not run on the FastAPI EventLoop.
`SemanticRoutingExecutor` is a small adapter owned by `SemanticRouter`:

```text
ThreadPoolExecutor(max_workers=1)
+ asyncio.BoundedSemaphore(value=9)  # 1 running + at most 8 admitted
+ asyncio.wait_for(..., timeout=150 ms)
```

Defaults:

```text
workers = 1
queue capacity = 8
admission timeout = 5 ms
route timeout = 150 ms
```

The semaphore is acquired before submission, so the executor's internal
unbounded queue never receives more than the configured admitted capacity.
An admission timeout produces `UNKNOWN` with reason `semantic_overloaded`.

`asyncio.wait_for(asyncio.shield(future), route_timeout)` limits caller wait,
but Python cannot stop an already running encoder thread. On timeout, the
semaphore permit is released only after the executor future actually finishes;
it must not be released immediately and allow hidden overcommit.

Shutdown stops new admissions and invokes
`executor.shutdown(wait=True, cancel_futures=True)` outside the EventLoop.

This adapter is deliberately not generalized for model inference, database,
Tools, or other workloads.

### Hybrid router

`HybridRouter.route()` is asynchronous because only the semantic branch may
offload work:

```python
async def route(self, message: str) -> RoutingDecision:
    fast = self._rule.route(message)
    if fast is not None:
        return fast
    if self._semantic is None:
        return unknown("semantic_disabled")
    return await self._semantic.route(message)
```

Semantic initialization or runtime failure degrades to the rule router plus
`UNKNOWN` when routing mode is `hybrid_optional`. It fails startup only in
`hybrid_required` mode.

## Configuration

```text
TREESEM_AGENT_ROUTING_MODE=rule
TREESEM_AGENT_TASK_REGISTRY=<trusted local tasks.yaml>
TREESEM_AGENT_ROUTING_MODEL=<local model id/path>
TREESEM_AGENT_ROUTING_MODEL_REVISION=<pinned revision>
TREESEM_AGENT_ROUTING_WORKERS=1
TREESEM_AGENT_ROUTING_QUEUE_CAPACITY=8
TREESEM_AGENT_ROUTING_ADMISSION_TIMEOUT_MS=5
TREESEM_AGENT_ROUTING_TIMEOUT_MS=150
```

Supported modes:

| Mode | Behavior |
|---|---|
| `rule` | Rule fast path, then `UNKNOWN`; no embedding dependency |
| `hybrid_optional` | Semantic fallback enabled; load/runtime failure degrades to `UNKNOWN` |
| `hybrid_required` | Semantic model/config failure prevents startup |

`rule` remains the default until the calibration and promotion gate passes.
After promotion, local/demo configuration may default to `hybrid_optional`;
production-like deployments retain an explicit choice.

No runtime model download is permitted in the reproducible demo. Model cache
or local model path is mounted read-only. The Agent image size, cold-start time,
and RSS delta are recorded before promotion because the Agent currently does
not depend on Torch or Sentence Transformers.

## Observability

Low-cardinality Metrics:

```text
treesem_agent_routing_total{source,scope}
treesem_agent_routing_fallback_total{reason}
treesem_agent_routing_duration_seconds{source}
treesem_agent_routing_admitted
treesem_agent_routing_overloaded_total
```

Trace fields may include:

```text
scope
source
similarity_score
margin
secondary_score
reason
duration_ms
```

Similarity values are Trace data, not Metric labels. Raw queries and all
resource identifiers remain excluded.

Health remains a process/config check. Readiness reports the configured routing
mode and whether semantic routing is available, but `hybrid_optional` semantic
degradation does not make the Agent unready because it retains the safe
`UNKNOWN` path.

## Evaluation

The versioned routing corpus contains approximately 150 manually reviewed,
synthetic cases:

```text
90 known single intents
20 unknown/general
20 compositional/multi-intent
20 safety/refusal
```

Registry examples, calibration cases, and held-out cases must be disjoint and
deduplicated. Safety results are reported separately and are excluded from
ordinary intent Macro F1.

Reports include:

- known-intent exact-route accuracy;
- Macro F1;
- rule fast-path precision and coverage;
- unknown recall;
- compositional fallback recall;
- confusion matrix;
- p50/p95 route latency;
- semantic overload/timeout degradation;
- Agent image size, startup time, and RSS delta.

Threshold selection order:

1. zero safety regression;
2. minimize false deterministic routing;
3. achieve at least 95% unknown/compositional fallback recall when possible;
4. maximize known-intent Macro F1;
5. choose stricter thresholds on ties.

Semantic routing is enabled by default only if held-out results improve known
intent routing, do not materially worsen false deterministic routing, preserve
Agent safety, and achieve warm local p95 routing latency at or below 100 ms on
the recorded machine. Otherwise it remains feature-gated and the negative
result is documented.

## Testing Strategy

Unit tests use a fake `EmbeddingProvider`; they never download a model.

Coverage includes:

- Registry schema and cross-field validation;
- complex Rule Router precedence;
- semantic score, margin, and secondary-intent gates;
- semantic inability to produce Skill/refusal scopes;
- admission overload, route timeout, late future completion, and safe permit
  release;
- optional degradation and required-mode startup failure;
- Skill routing preservation;
- UNKNOWN Tool restriction;
- existing deterministic Workflow, bounded repair, RBAC/Capability, RAG, and
  64-case Agent evaluation regressions.

Real-model execution follows the existing promotion ladder:

```text
unit tests
  -> fake embedding routing evaluation
  -> real embedding calibration/held-out evaluation
  -> deterministic full Agent evaluation
  -> targeted real-LLM cases
  -> full real-LLM evaluation only if targeted cases pass
```

## Non-goals

- no general Planner or Replanner;
- no DAG/Workflow engine;
- no semantic safety decision replacing deterministic policy;
- no FAISS for a tiny intent example set;
- no generic scheduler abstraction;
- no parallel Tool execution;
- no dynamic Task Registry updates or remote task installation;
- no semantic Skill selection in this phase;
- no claim that one real-LLM run proves stable 100% quality.

## Completion Criteria

The upgrade is complete only when:

1. current rule and Skill behavior remains compatible;
2. Registry and routing failures are fail-fast or safely degraded according to
   mode;
3. EventLoop remains responsive while embedding is slow;
4. executor admissions and memory remain bounded under overload;
5. semantic routing cannot expand Tool permissions;
6. calibration and held-out reports are reproducible;
7. the promotion decision is supported by measured routing and resource data;
8. all existing Agent and project verification gates pass.
