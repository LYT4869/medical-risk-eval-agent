# treeSem Agent Hybrid Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current mixed keyword classifier with a safety-first, rule-fast-path, semantic-fallback router that improves business-intent coverage without blocking the FastAPI EventLoop or expanding Tool permissions.

**Architecture:** Deterministic safety checks run first. Explicit Skill requests and high-precision business rules run synchronously. Only rule misses enter an optional semantic router. Blocking embedding work is isolated by one fixed-size `ThreadPoolExecutor`, one bounded admission semaphore, and request deadlines. Ambiguous, overloaded, timed-out, or unavailable semantic routing degrades to the existing guarded `UNKNOWN` Open Agent path.

**Tech Stack:** Python 3.10, FastAPI, asyncio, `concurrent.futures.ThreadPoolExecutor`, PyYAML, Pydantic, sentence-transformers, unittest, existing treeSem Trace/Metrics.

**Spec:** `docs/superpowers/specs/2026-09-07-agent-hybrid-routing-design.md`

## Global constraints

- Do not change RBAC, Capability JWT, Tool Schema, Tool implementations, or medical refusal semantics.
- Semantic routing may return only `prediction`, `summary`, `explanation`, `history`, `comparison`, or `knowledge`.
- `skill`, `security_abuse`, and `medical_refusal` remain deterministic-only.
- Do not execute embeddings on the FastAPI EventLoop.
- Do not create a generic scheduler, custom thread-pool framework, Planner/Replanner, DAG engine, or FAISS intent index.
- Do not download the embedding model at service runtime.
- Do not promote hybrid routing by intuition. Keep `rule` as the default until held-out routing and Agent regressions pass.
- Every implementation task starts with a failing test and ends with a focused local commit.

---

## Task 1: Freeze the routing corpus and current baseline

**Files:**

- Create: `PythonServices/TreeSemAgent/evaluation/routing_cases.json`
- Create: `PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_evaluation.py`
- Read: `PythonServices/TreeSemAgent/evaluation/cases.json`
- Read: `PythonServices/TreeSemAgent/agent/run_guard.py`

- [ ] **Step 1: Add a failing corpus validation test**

Validate that each case contains a stable `case_id`, `split`, `category`, `message`, and `expected_scope`; IDs are unique; messages are normalized and duplicate-free across registry examples, calibration, and held-out splits.

```python
def test_routing_corpus_is_unique_and_split_safe(self):
    cases = load_cases(ROUTING_CASES)
    self.assertGreaterEqual(len(cases), 150)
    self.assertEqual(len({case.case_id for case in cases}), len(cases))
    assert_no_normalized_message_overlap(cases)
```

- [ ] **Step 2: Confirm the test fails because the corpus does not exist**

Run:

```bash
python -m unittest PythonServices.TreeSemAgent.tests.test_routing_evaluation -v
```

Expected: failure on missing corpus/loader.

- [ ] **Step 3: Add approximately 150 synthetic, manually reviewed cases**

Use these fixed categories:

```text
90 known single-intent cases, 15 per business scope
20 unknown/general cases
20 compositional or multi-intent cases
20 safety/refusal cases
```

Label safety cases separately so they do not inflate business-intent Macro F1. Do not use real patient text.

- [ ] **Step 4: Implement the baseline evaluator**

The evaluator must report:

```text
known exact-route accuracy
business Macro F1
rule precision and coverage
unknown recall
compositional fallback recall
safety accuracy
confusion matrix
p50/p95 route latency
```

It must support a pure function/callable router so future rule and semantic implementations use the same runner.

- [ ] **Step 5: Record the current keyword baseline**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
  --router current-rule --split held_out
```

Write the generated report under the repository's ignored local evaluation-output directory. The committed corpus is the contract; machine-specific result JSON is evidence, not source code.

- [ ] **Step 6: Run tests and commit**

```bash
python -m unittest PythonServices.TreeSemAgent.tests.test_routing_evaluation -v
git diff --check
git add PythonServices/TreeSemAgent/evaluation PythonServices/TreeSemAgent/tests/test_routing_evaluation.py
git commit -m "test: add agent routing evaluation corpus"
```

---

## Task 2: Introduce routing types and a validated Task Registry

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/routing_types.py`
- Create: `PythonServices/TreeSemAgent/agent/task_registry.py`
- Create: `PythonServices/TreeSemAgent/config/tasks.yaml`
- Create: `PythonServices/TreeSemAgent/tests/test_task_registry.py`
- Modify: `PythonServices/TreeSemAgent/agent/run_guard.py`
- Modify: `PythonServices/TreeSemAgent/agent/workflow.py`

- [ ] **Step 1: Add failing Registry validation tests**

Cover valid loading plus duplicate scope, unknown key, unknown Tool, unsupported mode, missing semantic examples, safety scope in YAML, and Workflow stage outside `allowed_tools`.

```python
def test_registry_rejects_stage_outside_allowed_tools(self):
    registry = registry_from_yaml("""
    tasks:
      - scope: prediction
        mode: deterministic
        allowed_tools: [get_prediction]
        workflow_stages: [predict_sample]
        intent_examples: ["预测演示样本"]
    """)
    with self.assertRaises(TaskRegistryError):
        registry.load()
```

- [ ] **Step 2: Run the new tests and verify failure**

```bash
python -m unittest PythonServices.TreeSemAgent.tests.test_task_registry -v
```

- [ ] **Step 3: Move shared routing types out of `run_guard.py`**

Define dependency-neutral types:

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
    secondary_score: float | None = None
    reason: str | None = None
```

Update imports atomically so existing tests never see two `RequestScope` enums.

- [ ] **Step 4: Implement strict `TaskRegistry` loading**

The registry owns only static task metadata:

```text
scope, mode, rule_terms, intent_examples, workflow_stages, allowed_tools
```

It must return immutable definitions and reject malformed files at startup. It is not a general expression language.

- [ ] **Step 5: Add six business task definitions**

Encode the current Tool permissions and deterministic stages for prediction, summary, explanation, history, comparison, and knowledge. Preserve complex precedence in Python code rather than YAML.

- [ ] **Step 6: Adapt `WorkflowPlanner` to Registry-backed stages**

Keep the existing trusted Skill plan separate. For registered business scopes, construct the plan from the Registry. For `UNKNOWN`, keep `OPEN_AGENT`.

- [ ] **Step 7: Run focused and existing tests, then commit**

```bash
python -m unittest \
  PythonServices.TreeSemAgent.tests.test_task_registry \
  PythonServices.TreeSemAgent.tests.test_run_guard \
  PythonServices.TreeSemAgent.tests.test_workflow -v
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/config PythonServices/TreeSemAgent/tests
git commit -m "refactor: centralize agent task metadata"
```

---

## Task 3: Separate the deterministic Safety Gate and Rule Router

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/routing.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing.py`
- Modify: `PythonServices/TreeSemAgent/agent/run_guard.py`
- Modify: `PythonServices/TreeSemAgent/agent/loop.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_run_guard.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_workflow.py`

- [ ] **Step 1: Add failing precedence and safety tests**

Required cases include:

- explicit Skill before business routing;
- comparison before history;
- stored prediction explanation before general knowledge;
- summary only when a stored/read context is present;
- “不要编造” does not trigger abuse refusal;
- unsafe personalized diagnosis and fabricated-source requests still refuse.

- [ ] **Step 2: Implement `SafetyGate`**

```python
@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    refusal_scope: RequestScope | None = None
    reason: str | None = None

class SafetyGate:
    def evaluate(self, message: str) -> SafetyDecision: ...
```

Move only deterministic safety/refusal recognition here.

- [ ] **Step 3: Implement `RuleRouter`**

`RuleRouter.route(message)` returns `RoutingDecision | None`. A miss must be `None`, not `UNKNOWN`, so the semantic fallback can distinguish “no rule matched” from a final decision.

Keep compound predicates in code; use Registry `rule_terms` only for simple markers.

- [ ] **Step 4: Make `AgentRunGuard` consume an already-selected scope**

Replace business classification with:

```python
@classmethod
def for_scope(cls, scope: RequestScope, *, include_summary: bool = False): ...
```

Retain Tool narrowing, successful-call suppression, knowledge limits, prediction side-effect protection, and Skill intersection unchanged.

- [ ] **Step 5: Integrate deterministic routing into `AgentLoop`**

Temporarily use `SafetyGate + RuleRouter`, with rule misses converted to `UNKNOWN`. This preserves behavior before semantic routing exists.

- [ ] **Step 6: Run regression tests and baseline comparison**

```bash
python -m unittest discover -s PythonServices/TreeSemAgent/tests -v
PYTHONPATH=PythonServices/TreeSemAgent \
python PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
  --router rule --split held_out
```

Safety accuracy and existing 64 Agent cases must not regress.

- [ ] **Step 7: Commit**

```bash
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/tests
git commit -m "refactor: separate agent safety and rule routing"
```

---

## Task 4: Implement semantic routing as pure, testable logic

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/semantic_routing.py`
- Create: `PythonServices/TreeSemAgent/tests/test_semantic_routing.py`
- Modify: `PythonServices/TreeSemAgent/agent/routing.py`

- [ ] **Step 1: Write failing tests with a fake embedding provider**

Test:

- clear top-1 match;
- score below minimum;
- top-1/top-2 margin too small;
- secondary score above compositional threshold;
- non-finite or wrong-sized vectors;
- inability to emit Skill or refusal scopes;
- immutable startup example cache.

```python
class FakeEmbeddingProvider:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]
```

- [ ] **Step 2: Define the provider boundary**

```python
class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
```

No sentence-transformers import belongs in the pure scoring code.

- [ ] **Step 3: Implement normalized cosine routing**

At construction, embed and L2-normalize Registry examples. At request time:

```text
scope score = maximum cosine similarity among that scope's examples
margin = top1 - top2
```

Return `UNKNOWN` when any threshold gate fails. Name these values similarity scores, not probabilities.

- [ ] **Step 4: Implement asynchronous `HybridRouter` composition**

```python
async def route(self, message: str) -> RoutingDecision:
    fast = self._rules.route(message)
    if fast is not None:
        return fast
    if self._semantic is None:
        return unknown("semantic_disabled")
    return await self._semantic.route(message)
```

This task uses a fake async semantic adapter; bounded thread execution comes next.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest \
  PythonServices.TreeSemAgent.tests.test_semantic_routing \
  PythonServices.TreeSemAgent.tests.test_routing -v
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/tests
git commit -m "feat: add evaluated semantic intent routing"
```

---

## Task 5: Add the lightweight bounded blocking adapter

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/routing_executor.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_executor.py`
- Modify: `PythonServices/TreeSemAgent/agent/semantic_routing.py`

- [ ] **Step 1: Write failing concurrency tests**

Cover:

- EventLoop heartbeat continues while a fake encoder blocks;
- at most `workers + queue_capacity` calls are admitted;
- admission timeout returns `semantic_overloaded`;
- route deadline returns `semantic_timeout`;
- a timed-out running call retains its permit until the worker future completes;
- shutdown rejects new work and finishes without blocking the EventLoop.

The late-completion test is mandatory:

```python
async def test_timeout_does_not_release_permit_before_worker_finishes(self):
    first = asyncio.create_task(executor.run(block_until_released))
    await assert_times_out(first)
    second = await executor.try_run(noop)
    self.assertEqual(second.reason, "semantic_overloaded")
    release_worker.set()
    await executor.wait_until_idle()
```

- [ ] **Step 2: Implement one fixed executor and one admission semaphore**

Defaults:

```text
max_workers = 1
queue_capacity = 8
admission_timeout = 5 ms
route_timeout = 150 ms
```

Use `asyncio.BoundedSemaphore(workers + queue_capacity)` before submitting to prevent the standard executor's internal unbounded queue from growing.

- [ ] **Step 3: Implement correct timeout ownership**

Use:

```python
wrapped = asyncio.wrap_future(pool.submit(fn))
try:
    return await asyncio.wait_for(asyncio.shield(wrapped), timeout)
finally:
    if wrapped.done():
        semaphore.release()
    else:
        wrapped.add_done_callback(release_permit_thread_safely)
```

The exact implementation must make permit release one-shot and EventLoop-safe. Do not release the permit in the caller timeout path while the worker is still running.

- [ ] **Step 4: Add non-blocking shutdown**

Mark the adapter closed, reject new admissions, and run `executor.shutdown(wait=True, cancel_futures=True)` through an external helper thread or `asyncio.to_thread`; do not invoke blocking shutdown on FastAPI's EventLoop.

- [ ] **Step 5: Connect `SemanticRouter` to the adapter**

All provider encoding must pass through this adapter. Optional-mode overload, timeout, or runtime error becomes an `UNKNOWN` decision with a stable reason.

- [ ] **Step 6: Run the concurrency test repeatedly**

```bash
for run in $(seq 1 20); do
  python -m unittest PythonServices.TreeSemAgent.tests.test_routing_executor -q || exit 1
done
python -m unittest PythonServices.TreeSemAgent.tests.test_semantic_routing -v
```

- [ ] **Step 7: Commit**

```bash
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/tests
git commit -m "feat: bound semantic routing execution"
```

---

## Task 6: Add the real embedding provider and calibrate thresholds

**Files:**

- Create: `PythonServices/TreeSemAgent/agent/embedding_provider.py`
- Create: `PythonServices/TreeSemAgent/requirements-routing.txt`
- Create: `PythonServices/TreeSemAgent/config/routing_thresholds.json`
- Create: `PythonServices/TreeSemAgent/evaluation/calibrate_routing.py`
- Create: `PythonServices/TreeSemAgent/tests/test_embedding_provider.py`
- Modify: `PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py`
- Modify: `PythonServices/TreeSemAgent/config/tasks.yaml`

- [ ] **Step 1: Add failing provider and configuration tests**

Test local-only loading, pinned revision presence, vector validation, unavailable optional dependency, and deterministic threshold file validation. Unit tests mock the model; they do not download weights.

- [ ] **Step 2: Pin routing dependencies separately**

Use the same compatible versions already proven by the Knowledge service:

```text
sentence-transformers==5.7.0
transformers==4.57.6
```

Keep these in `requirements-routing.txt` until promotion, so default Agent development does not silently gain a large Torch dependency.

- [ ] **Step 3: Implement local-only sentence-transformers loading**

Candidate model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Resolve and record its exact upstream revision in configuration during this task. Load from the configured local path/cache with network access disabled in the runtime path.

- [ ] **Step 4: Add calibration search**

Sweep similarity, margin, and secondary-intent thresholds on the calibration split. Select by this fixed order:

1. zero safety regression;
2. minimize false deterministic routing;
3. maximize unknown/compositional fallback recall, targeting at least 95%;
4. maximize known-intent Macro F1;
5. choose stricter thresholds on ties.

Write the selected numeric thresholds and corpus checksum to `routing_thresholds.json`; never hard-code invented scores.

- [ ] **Step 5: Evaluate once on held-out cases**

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python PythonServices/TreeSemAgent/evaluation/calibrate_routing.py \
  --cases PythonServices/TreeSemAgent/evaluation/routing_cases.json \
  --tasks PythonServices/TreeSemAgent/config/tasks.yaml \
  --output PythonServices/TreeSemAgent/config/routing_thresholds.json

PYTHONPATH=PythonServices/TreeSemAgent \
python PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
  --router hybrid --split held_out
```

Do not revisit thresholds based on held-out errors without creating a new corpus version and a fresh held-out split.

- [ ] **Step 6: Record resource evidence**

Measure warm p50/p95 route time, cold startup time, process RSS delta, and dependency/image-size delta. This evidence determines whether hybrid routing is promoted or remains optional.

- [ ] **Step 7: Run tests and commit**

```bash
python -m unittest \
  PythonServices.TreeSemAgent.tests.test_embedding_provider \
  PythonServices.TreeSemAgent.tests.test_routing_evaluation -v
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/config \
  PythonServices/TreeSemAgent/evaluation PythonServices/TreeSemAgent/requirements-routing.txt \
  PythonServices/TreeSemAgent/tests
git commit -m "test: calibrate semantic routing thresholds"
```

---

## Task 7: Integrate routing modes into Agent startup and lifecycle

**Files:**

- Modify: `PythonServices/TreeSemAgent/agent/loop.py`
- Modify: `PythonServices/TreeSemAgent/server.py`
- Modify: `PythonServices/TreeSemAgent/agent/observability.py`
- Create: `PythonServices/TreeSemAgent/tests/test_server_routing.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_agent_loop.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_observability.py`

- [ ] **Step 1: Add failing mode and lifecycle tests**

Cover:

- `rule` never constructs an embedding provider;
- `hybrid_optional` degrades when model/config/runtime fails;
- `hybrid_required` fails startup when semantic routing is unavailable;
- app shutdown closes the routing executor;
- injected test `AgentLoop` remains supported by `create_app(loop=...)`;
- `/health` and `/ready` expose low-risk routing state.

- [ ] **Step 2: Inject a router into `AgentLoop`**

The run sequence becomes:

```text
SafetyGate
  -> refusal or HybridRouter
  -> AgentRunGuard.for_scope
  -> WorkflowPlanner
  -> existing Agent Loop
```

Rule-hit latency must not enter the executor or embedding provider.

- [ ] **Step 3: Add validated environment configuration**

```text
TREESEM_AGENT_ROUTING_MODE=rule
TREESEM_AGENT_TASK_REGISTRY=<tasks.yaml>
TREESEM_AGENT_ROUTING_MODEL=<local id/path>
TREESEM_AGENT_ROUTING_MODEL_REVISION=<exact revision>
TREESEM_AGENT_ROUTING_WORKERS=1
TREESEM_AGENT_ROUTING_QUEUE_CAPACITY=8
TREESEM_AGENT_ROUTING_ADMISSION_TIMEOUT_MS=5
TREESEM_AGENT_ROUTING_TIMEOUT_MS=150
```

Reject non-positive limits, unknown mode, and inconsistent model settings at startup.

- [ ] **Step 4: Add readiness semantics**

`hybrid_optional` remains ready when semantic routing degrades because the guarded `UNKNOWN` path is still available. `hybrid_required` cannot become ready without a loaded model. Health must not perform an embedding request.

- [ ] **Step 5: Add routing Metrics and Trace fields**

Metrics must use only bounded labels:

```text
treesem_agent_routing_total{source,scope}
treesem_agent_routing_fallback_total{reason}
treesem_agent_routing_duration_seconds{source}
treesem_agent_routing_admitted
treesem_agent_routing_overloaded_total
```

Similarity, margin, and secondary score may be Trace fields but never labels. Raw messages must not be logged.

- [ ] **Step 6: Run Agent regressions**

```bash
python -m unittest discover -s PythonServices/TreeSemAgent/tests -v
PYTHONPATH=PythonServices/TreeSemAgent \
python PythonServices/TreeSemAgent/evaluation/run_evaluation.py \
  --mode deterministic
```

The existing deterministic 64-case gate must remain 100%.

- [ ] **Step 7: Commit**

```bash
git diff --check
git add PythonServices/TreeSemAgent/agent PythonServices/TreeSemAgent/server.py \
  PythonServices/TreeSemAgent/tests
git commit -m "feat: integrate safe hybrid agent routing"
```

---

## Task 8: Package semantic routing without burdening the default path

**Files:**

- Modify: `deploy/docker/agent.Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `Makefile`
- Create: `scripts/prepare-routing-model.sh`
- Create: `docs/reports/agent-routing-evaluation.md`

- [ ] **Step 1: Add failing static/deployment checks**

Extend the existing verification entry to assert:

- model revision is pinned;
- runtime model download is disabled;
- semantic model cache is read-only;
- no API keys or model cache files enter the image/Git;
- default `rule` mode still builds without optional routing dependencies.

- [ ] **Step 2: Add an optional Docker build target or build argument**

Keep the base Agent image on `requirements.txt`. Install `requirements-routing.txt` only for the semantic-routing image target/build argument. Do not duplicate the whole Dockerfile.

- [ ] **Step 3: Mount the routing model cache read-only**

Only hybrid deployments receive the mount and local-only environment settings. `prepare-routing-model.sh` performs an explicit operator-invoked download/checksum preparation step; service startup never downloads.

- [ ] **Step 4: Add reproducible commands**

Provide Make targets for:

```text
routing-unit
routing-calibrate
routing-evaluate
routing-load-smoke
```

The smoke test drives concurrent semantic misses and verifies bounded overload plus responsive `/health`.

- [ ] **Step 5: Write the promotion report**

Record corpus version, model/revision, thresholds, rule baseline, hybrid held-out metrics, latency, RSS, cold start, dependency/image-size delta, and the promotion decision.

- [ ] **Step 6: Commit**

```bash
make routing-unit
docker compose config >/dev/null
git diff --check
git add deploy/docker/agent.Dockerfile docker-compose.yml .env.example Makefile \
  scripts/prepare-routing-model.sh docs/reports/agent-routing-evaluation.md
git commit -m "build: package optional semantic agent routing"
```

---

## Task 9: Run the promotion gate and complete documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/agent-llm-optimization-interview-case.md`
- Modify: `docs/treesem-interview-guide.md`
- Modify: `docs/reports/agent-routing-evaluation.md`
- Modify: `PythonServices/TreeSemAgent/evaluation/cases.json` only if new regression cases are added before the final held-out run; otherwise leave it unchanged.

- [ ] **Step 1: Run the complete local verification gate**

```bash
make verify
make routing-evaluate
```

Run routing executor concurrency tests 20 times. Confirm the EventLoop remains responsive and admitted work never exceeds the configured bound.

- [ ] **Step 2: Apply the promotion criteria**

Promote `hybrid_optional` as the demo/local default only when all are true:

```text
safety regression = 0
known-intent routing improves over rules
false deterministic routing does not materially worsen
unknown/compositional fallback recall reaches 95% when the corpus permits
warm p95 routing latency <= 100 ms on the recorded machine
64-case deterministic Agent evaluation = 100%
```

If any gate fails, retain `rule` as the default, keep semantic routing feature-gated, and document the negative result. This is a valid outcome.

- [ ] **Step 3: Run targeted real-LLM regression**

Use only cases whose intended route changed plus safety, ambiguity, Skill, and multi-intent cases. Verify Tool choice, grounding, citations, step count, and latency. Run the full paid real-LLM suite only if the targeted set is clean.

- [ ] **Step 4: Update the interview narrative**

Explain the project decision in interview language:

```text
rules remain a high-precision fast path;
semantic routing improves paraphrase coverage;
ambiguous requests fall back to guarded Open Agent;
semantic routing chooses workflow but never grants permission;
blocking embeddings use a tiny bounded isolation adapter;
promotion is based on held-out quality and resource evidence.
```

Do not claim that keyword routing was removed, that embeddings are probabilities, or that one real-model run proves universal quality.

- [ ] **Step 5: Final verification and commit**

```bash
python -m unittest discover -s PythonServices/TreeSemAgent/tests -v
make verify
git diff --check
git status --short
git add README.md docs PythonServices/TreeSemAgent/evaluation/cases.json
git commit -m "docs: document agent routing evaluation and tradeoffs"
```

Do not push until the user reviews the reports and commits.

---

## Final acceptance checklist

- [ ] Safety and medical refusals remain deterministic and unchanged.
- [ ] Explicit Skill routing remains deterministic and precedes semantic business routing.
- [ ] Rule hits perform no embedding and no executor submission.
- [ ] Semantic routing cannot produce safety/Skill scopes or expand allowed Tools.
- [ ] Ambiguity, overload, timeout, and optional dependency failures produce safe `UNKNOWN` degradation.
- [ ] `ThreadPoolExecutor`, admission semaphore, and deadline are fixed and bounded.
- [ ] Timed-out worker permits are released only after the worker actually completes.
- [ ] EventLoop responsiveness and 20-round concurrency regression pass.
- [ ] Routing corpus, calibration, held-out report, and resource measurements are reproducible.
- [ ] Existing 64-case deterministic Agent evaluation remains 100%.
- [ ] Real-LLM targeted regression passes before any full paid rerun.
- [ ] Default-mode promotion follows measured gates; a failed promotion leaves the current safe default intact.
- [ ] Documentation and interview material describe the architecture without referring to implementation-level function names.
