# treeSem Agent Rule Taxonomy and Structured Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize the existing keyword-assisted Rule Router as typed action/object/reference evidence plus a locked business taxonomy, while preserving deterministic Workflows, guarded Open Agent fallback, and the optional E5 candidate.

**Architecture:** Split the current `RuleRouter.route()` logic into a pure `RuleEvidenceExtractor` and a pure `RuleIntentResolver`; retain `RuleRouter` as the public facade. Add an explicit taxonomy contract and additive evaluation metrics that distinguish safe abstention from harmful misrouting. Safety, Hybrid ONNX routing, Workflow execution, Tool permissions, and external APIs remain structurally unchanged.

**Tech Stack:** Python 3.10, dataclasses, Enum, unittest, YAML Task Registry, existing Agent evaluation scripts

**Spec:** `docs/superpowers/specs/2026-09-09-agent-rule-taxonomy-design.md`

## Global Constraints

- Do not modify the treeSem clinical model, Serving Bundle, model output, or historical `trivae` training code.
- Do not replace or retune `intfloat/multilingual-e5-small`; do not regenerate the frozen FP32 routing Artifact in this cycle.
- Keep `TREESEM_AGENT_ROUTING_MODE=rule` as the default.
- Keep `rule`, `hybrid_optional`, and `hybrid_required` behavior and configuration compatible.
- Do not change deterministic Workflow stages, Tool implementations, RBAC, Capability JWTs, or Skill loading.
- Do not add a learned classifier, LLM Router, rule DSL, routing microservice, or new executor.
- Do not edit the frozen 120/240 Routing Quality Set or use its Heldout split for tuning or promotion claims.
- SafetyPolicy remains separate and executes before business routing; this cycle does not expand its lexical risk coverage.
- Follow TDD, keep every commit independently testable, and do not commit generated evaluation artifacts.

---

### Task 1: Freeze the business-intent taxonomy as an executable contract

**Files:**
- Create: `docs/agent-routing-taxonomy.md`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_taxonomy_contract.py`
- Modify: `docs/roadmap.md`

**Interfaces:**
- Consumes: existing `RequestScope`, `RuleRouter.route()`, and `WorkflowPlanner.for_request()`.
- Produces: a reviewed scope/dependency contract and canonical tests that later refactoring must satisfy without depending on implementation details.

- [ ] **Step 1: Write the taxonomy document**

Document the six scope definitions and the following dependency matrix verbatim:

```text
prediction  : a new demo/sample evaluation; no absorbed business scope
summary     : stored prediction facts; requires a stored-record reference
explanation : explanation of one stored prediction; may absorb summary for that record
history     : list saved prediction records
comparison  : compare multiple saved predictions; may absorb history and summary for those records
knowledge   : general model/PPH evidence retrieval; never absorbed into a patient-record task
```

Document these independent compositions as `UNKNOWN`:

```text
prediction + explanation
prediction + history
explanation + comparison
explanation + knowledge
comparison + knowledge
history + knowledge
three or more independent scopes
```

- [ ] **Step 2: Write failing contract tests before changing routing code**

Use table-driven tests with at least these exact cases:

```python
DEPENDENCY_CASES = (
    ("比较最近两次预测", RequestScope.COMPARISON, False),
    ("给出最新置信度，并与上一条记录比较", RequestScope.COMPARISON, False),
    ("show history，再 compare 最新两条", RequestScope.COMPARISON, False),
    ("解释当前预测的标签、概率和决策路径", RequestScope.EXPLANATION, True),
)

COMPOSITIONAL_CASES = (
    "运行样本 3，然后解释刚生成的结果",
    "解释当前预测，再查找产后出血指南",
    "比较最近两次预测，并检索模型限制",
    "列出历史记录，然后预测演示样本 5",
    "比较结果并分别解释两条决策路径",
)

AMBIGUOUS_CASES = (
    "帮我看看这个",
    "处理一下当前情况",
    "Can you help with this?",
)
```

Assert dependency cases select one expected scope, explanation records
`include_summary=True`, and compositional/ambiguous cases resolve to
`RequestScope.UNKNOWN` with stable reasons.

- [ ] **Step 3: Run the focused contract test and capture the baseline failures**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_routing_taxonomy_contract -v
```

Expected: existing correct cases pass; newly locked expressions that expose
mixed extraction/resolution behavior fail before the refactor. Record only
case IDs and expected/actual scopes, not user or patient data.

- [ ] **Step 4: Add a roadmap entry without claiming a quality improvement**

Add an in-progress entry stating that the work formalizes Rule evidence and
taxonomy. Explicitly state that the historical 240 Heldout result remains
unchanged and no new promotion claim exists.

- [ ] **Step 5: Commit the contract checkpoint**

```bash
git add docs/agent-routing-taxonomy.md docs/roadmap.md \
  PythonServices/TreeSemAgent/tests/test_routing_taxonomy_contract.py \
  docs/superpowers/specs/2026-09-09-agent-rule-taxonomy-design.md \
  docs/superpowers/plans/2026-09-09-agent-rule-taxonomy.md
git commit -m "test: freeze agent routing taxonomy contract"
```

---

### Task 2: Introduce typed rule evidence extraction

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/rule_evidence.py`
- Create: `PythonServices/TreeSemAgent/tests/test_rule_evidence.py`

**Interfaces:**
- Consumes: normalized user text and `TaskRegistry.definition(scope).rule_terms`.
- Produces:

```python
class IntentAction(str, Enum): ...
class IntentObject(str, Enum): ...
class IntentReference(str, Enum): ...

@dataclass(frozen=True)
class RuleEvidence:
    actions: frozenset[IntentAction]
    objects: frozenset[IntentObject]
    references: frozenset[IntentReference]
    registry_scopes: frozenset[RequestScope]
    explicit_skill: bool = False
    vague_reference: bool = False

class RuleEvidenceExtractor:
    def __init__(self, registry: TaskRegistry | None = None): ...
    def extract(self, message: str) -> RuleEvidence: ...
```

- [ ] **Step 1: Write failing extraction tests**

Cover these independent facts:

```python
"运行第 3 个演示样本"
  actions={PREDICT}
  objects={DEMO_SAMPLE}
  references={EXPLICIT_SAMPLE}

"查看当前预测的概率"
  actions={READ}
  objects={PREDICTION_FACT}
  references={CURRENT_PREDICTION}

"解释刚才结果的决策路径"
  actions={EXPLAIN}
  objects={EXPLANATION_DETAIL}
  references={CURRENT_PREDICTION}

"比较这次与上一次结果"
  actions={COMPARE}
  objects={PREDICTION_FACT}
  references={CURRENT_PREDICTION, PRIOR_PREDICTION, MULTIPLE_PREDICTIONS}

"查找产后出血指南"
  actions={RETRIEVE}
  objects={KNOWLEDGE}

"使用解释技能"
  explicit_skill=True
```

Also assert NFKC normalization, case folding, whitespace folding, and benign
unrelated text producing empty business evidence.

- [ ] **Step 2: Run the extraction tests and verify import failure**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_rule_evidence -v
```

Expected: FAIL because `agent.rule_evidence` does not exist.

- [ ] **Step 3: Implement the dependency-neutral evidence types**

Create the three enums and immutable `RuleEvidence` exactly as declared in
the Interfaces block. Keep marker collections private and grouped by semantic
role rather than by final scope:

```text
action markers    : predict/read/explain/list/compare/retrieve
object markers    : demo sample/prediction fact/explanation detail/history/knowledge
reference markers : explicit sample/current/prior/multiple predictions
control markers   : explicit skill/vague reference
```

Use one `_contains(message, markers)` helper and one NFKC `normalize()` helper.
Do not add stemming, third-party NLP libraries, or a configuration DSL.

- [ ] **Step 4: Implement `RuleEvidenceExtractor.extract()`**

Populate semantic sets from finite marker groups and add
`registry_scopes` when a Task Registry term matches. Registry matches are
coarse evidence only; they must not directly select a final scope.

Treat `当前预测`, `当前结果`, and `刚才结果` as current references; treat
`上一次`, `之前一条`, and `prior/previous prediction` as prior references;
derive `MULTIPLE_PREDICTIONS` when comparison language coexists with current
and prior references or explicit plural-record language.

- [ ] **Step 5: Run extraction and existing routing tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_rule_evidence \
  tests.test_routing -v
```

Expected: extraction tests PASS and existing routing behavior remains PASS;
the new extractor is not yet the routing decision source.

- [ ] **Step 6: Commit the typed extraction checkpoint**

```bash
git add PythonServices/TreeSemAgent/agent/rule_evidence.py \
  PythonServices/TreeSemAgent/tests/test_rule_evidence.py
git commit -m "refactor: extract typed agent routing evidence"
```

---

### Task 3: Resolve typed evidence through the locked taxonomy

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/rule_resolver.py`
- Create: `PythonServices/TreeSemAgent/tests/test_rule_resolver.py`
- Modify: `PythonServices/TreeSemAgent/agent/routing.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_routing.py`

**Interfaces:**
- Consumes: `RuleEvidence` from Task 2.
- Produces:

```python
class RuleIntentResolver:
    def resolve(self, evidence: RuleEvidence) -> RoutingDecision | None: ...
```

- [ ] **Step 1: Write failing resolver unit tests**

Construct `RuleEvidence` directly rather than passing text. Prove:

```text
PREDICT + DEMO_SAMPLE                         -> prediction
READ + PREDICTION_FACT + CURRENT_PREDICTION  -> summary
EXPLAIN + EXPLANATION_DETAIL + CURRENT       -> explanation
LIST + HISTORY                               -> history
COMPARE + MULTIPLE_PREDICTIONS               -> comparison
RETRIEVE + KNOWLEDGE                         -> knowledge
```

Prove dependency absorption:

```text
comparison + history/summary evidence -> comparison
explanation + summary evidence         -> explanation with include_summary
```

Prove independent combinations return `UNKNOWN` with
`reason="rule_compositional"`, vague references return `UNKNOWN` with
`reason="rule_ambiguous_reference"`, and incomplete evidence returns `None`.

- [ ] **Step 2: Run resolver tests and verify import failure**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_rule_resolver -v
```

Expected: FAIL because `agent.rule_resolver` does not exist.

- [ ] **Step 3: Implement candidate construction**

Implement one private predicate per business scope. Each predicate consumes
only typed evidence. Do not access raw strings inside the resolver.

Use the exact taxonomy requirements from the spec. A bare registry scope may
support a predicate but must not bypass a required stored reference for
`summary` or `explanation`.

- [ ] **Step 4: Implement dependency absorption and composition handling**

Apply these rules in order:

```python
if explicit_skill:
    return SKILL
if comparison is complete:
    discard dependency-only history and summary candidates
if explanation is complete:
    discard dependency-only summary candidate
if more than one independent candidate remains:
    return UNKNOWN(rule_compositional)
if one complete candidate remains:
    return that scope
if vague_reference:
    return UNKNOWN(rule_ambiguous_reference)
return None
```

Set `include_summary=True` only when the explanation request explicitly asks
for prediction facts from the same stored prediction.

- [ ] **Step 5: Turn `RuleRouter` into the compatibility facade**

Keep its public signature unchanged:

```python
class RuleRouter:
    def __init__(self, registry: TaskRegistry | None = None):
        self._extractor = RuleEvidenceExtractor(registry)
        self._resolver = RuleIntentResolver()

    def route(self, message: str) -> RoutingDecision | None:
        return self._resolver.resolve(self._extractor.extract(message))
```

Remove duplicated inline marker logic from `routing.py`; keep `RuleOnlyRouter`
and `HybridRouter` unchanged except for imports.

- [ ] **Step 6: Run resolver, routing, Workflow, and Guard tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_rule_resolver \
  tests.test_routing \
  tests.test_routing_taxonomy_contract \
  tests.test_workflow \
  tests.test_run_guard -v
```

Expected: all PASS. Confirm existing Workflow stages and allowed Tool sets are
unchanged for the same final scope.

- [ ] **Step 7: Commit the resolver checkpoint**

```bash
git add PythonServices/TreeSemAgent/agent/rule_resolver.py \
  PythonServices/TreeSemAgent/agent/routing.py \
  PythonServices/TreeSemAgent/tests/test_rule_resolver.py \
  PythonServices/TreeSemAgent/tests/test_routing.py \
  PythonServices/TreeSemAgent/tests/test_routing_taxonomy_contract.py
git commit -m "refactor: resolve agent intents through typed taxonomy"
```

---

### Task 4: Preserve Rule, Hybrid, Safety, and fallback boundaries

**Files:**
- Modify: `PythonServices/TreeSemAgent/tests/test_server_routing.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_semantic_routing.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_run_guard.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py`

**Interfaces:**
- Consumes: unchanged `RuleRouter.route()` facade and existing
  `HybridRouter`, `SemanticRouter`, `AgentRunGuard` interfaces.
- Produces: integration evidence that this refactor does not promote semantic
  routing, weaken safety, or alter failure degradation.

- [ ] **Step 1: Add mode-boundary tests**

Assert:

```text
rule mode does not construct an embedding provider
rule hit in Hybrid mode does not invoke semantic routing
rule miss in Hybrid mode invokes semantic routing exactly once
semantic low similarity returns UNKNOWN
semantic timeout/overload in optional mode returns UNKNOWN
hybrid_required still fails startup for invalid Artifact/configuration
```

- [ ] **Step 2: Add safety-order tests**

Patch the business router with a counting fake and prove that a deterministic
medical or security refusal never calls it. Prove that benign educational
questions continue past SafetyPolicy and remain subject to normal routing.

- [ ] **Step 3: Add Tool-boundary tests**

For each final scope, assert the initial Tool set still equals the Task
Registry definition. For `UNKNOWN`, assert the Open Agent receives only the
existing read-only native Tool set. Confirm routing decisions cannot add tools
outside Registry/RBAC/Capability intersections.

- [ ] **Step 4: Run the complete routing test target**

Run:

```bash
make routing-unit
make routing-load-smoke
```

Expected: all routing tests pass and the bounded executor passes 20/20 load
rounds. No model download occurs in `rule` mode.

- [ ] **Step 5: Commit the integration checkpoint**

```bash
git add PythonServices/TreeSemAgent/tests/test_server_routing.py \
  PythonServices/TreeSemAgent/tests/test_semantic_routing.py \
  PythonServices/TreeSemAgent/tests/test_run_guard.py \
  PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py
git commit -m "test: preserve guarded routing fallback boundaries"
```

---

### Task 5: Separate abstention, misrouting, and deterministic coverage metrics

**Files:**
- Modify: `PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_routing_evaluation.py`
- Modify: `docs/reports/agent-routing-evaluation.md`

**Interfaces:**
- Consumes: `RoutingCase`, expected scope, actual scope, and existing report
  metrics.
- Produces additive JSON report fields:

```text
deterministic_precision: float
deterministic_coverage: float
known_abstention_rate: float
known_misroute_rate: float
unknown_forced_route_rate: float
compositional_forced_route_rate: float
per_scope_precision_recall: dict[str, {precision: float, recall: float}]
```

- [ ] **Step 1: Write failing metric tests with a hand-calculated fixture**

Use a compact fixture containing:

```text
one correctly routed known case
one known case abstained to UNKNOWN
one known case sent to the wrong business Workflow
one Unknown case forced into a business Workflow
one correctly rejected Unknown case
one compositional case forced into a business Workflow
one correctly rejected compositional case
```

Assert each numerator and denominator explicitly. Also assert legacy
`rule_precision` and `rule_coverage` remain present and numerically equal to
the corresponding deterministic fields for Rule-only evaluation.

- [ ] **Step 2: Run the focused evaluation test and verify missing fields**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_routing_evaluation -v
```

Expected: FAIL because the additive report fields do not exist.

- [ ] **Step 3: Implement the additive counters**

Define:

```text
known abstention = known expected business scope, actual UNKNOWN
known misroute = known expected business scope, actual different business scope
Unknown forced route = unknown category, actual business scope
compositional forced route = compositional category, actual business scope
deterministic precision = correct business routes / all business routes
deterministic coverage = all business routes / all non-safety cases
```

Compute per-scope precision and recall only for the six business scopes.
Retain the current confusion matrix and latency fields.

- [ ] **Step 4: Update metric documentation without rewriting historical data**

Explain that old `rule_precision` is a legacy field name and is misleading in
Hybrid reports because a deterministic route can come from Rule or Semantic.
Add the 2026-09-09 Heldout baseline as immutable historical evidence:

```text
Rule known accuracy 18.33%
Rule business Macro F1 25.64%
Rule Unknown recall 97.50%
Rule legacy precision 37.29%
Rule coverage 29.50%

Hybrid known accuracy 22.50%
Hybrid business Macro F1 31.15%
Hybrid Unknown recall 85.00%
Hybrid legacy precision 38.57%
Hybrid coverage 35.00%
```

Do not rewrite `artifacts/evaluation/routing/closure-heldout-*.json`.

- [ ] **Step 5: Run evaluation tests and a non-promotional diagnostic report**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest \
  tests.test_routing_evaluation -v

PYTHONPATH=PythonServices/TreeSemAgent \
python3 PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
  --router rule \
  --split calibration \
  --cases PythonServices/TreeSemAgent/evaluation/routing_quality_set.json
```

Expected: tests PASS and the printed report contains all old and new fields.
Do not write the output into the frozen Artifact report directory.

- [ ] **Step 6: Commit the evaluation checkpoint**

```bash
git add PythonServices/TreeSemAgent/evaluation/run_routing_evaluation.py \
  PythonServices/TreeSemAgent/tests/test_routing_evaluation.py \
  docs/reports/agent-routing-evaluation.md
git commit -m "test: distinguish agent abstention from misrouting"
```

---

### Task 6: Complete regressions, documentation, and interview evidence

**Files:**
- Modify: `docs/agent-llm-optimization-interview-case.md`
- Modify: `docs/treesem-interview-guide.md`
- Modify: `docs/reports/agent-routing-safety-closure.md`
- Modify: `docs/roadmap.md`

**Interfaces:**
- Consumes: Tasks 1-5 implementation and verification evidence.
- Produces: final handoff material that distinguishes implementation success,
  routing-quality evidence, and unsupported claims.

- [ ] **Step 1: Run the complete Python Agent regression**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent \
python3 -m unittest discover \
  -s PythonServices/TreeSemAgent/tests -p 'test_*.py' -v
```

Expected: all available tests PASS; optional dependency skips must identify
their dependency and remain at the existing baseline or lower.

- [ ] **Step 2: Run Knowledge, C++, and Adapter regressions**

Run the repository's existing full verification entry points:

```bash
python3 -m unittest discover \
  -s PythonServices/TreeSemKnowledge/tests -p 'test_*.py' -v
ctest --test-dir build --output-on-failure
python3 -m unittest discover \
  -s PythonServices/TreeSemModelAdapter/tests -p 'test_*.py' -v
```

Expected: Knowledge, C++/cross-service, and Adapter suites PASS. If the active
build directory differs, resolve it from the documented build command rather
than creating an unrelated configuration.

- [ ] **Step 3: Run deterministic Agent evaluation**

Use the existing Fake LLM 64-scenario command documented in
`docs/m9-observability-evaluation.md`. Require:

```text
scenario task outcome = 64/64
prediction grounding validity = 100%
citation validity = 100%
cross-role leakage = 0
blocked/redundant Tool attempts = 0
```

This verifies protocol compatibility; it does not prove natural-language
Router generalization.

- [ ] **Step 4: Check repository hygiene**

Run:

```bash
git diff --check
git status --short
git ls-files | rg 'artifacts/evaluation/routing/closure-heldout|\.onnx$'
```

Expected: no whitespace errors, only intended source/docs changes before the
final commit, and no generated ONNX model or newly generated frozen evaluation
output staged for commit.

- [ ] **Step 5: Update interviewer-facing documentation**

Record these exact distinctions:

```text
The Router uses structured lexical evidence, not a learned NLU parser.
Rules optimize precision and latency, not complete language coverage.
Known abstention costs an Open Agent call; wrong deterministic routing can run the wrong Workflow.
Safety language detection is not the execution authorization boundary.
The E5 FP32 implementation remains a valid optional backend but its policy did not pass Heldout promotion.
No new generalization claim is made without a fresh independent Heldout set.
```

Add one interview question covering why Accuracy alone is insufficient for a
selective fast-path classifier, with Precision, Coverage, abstention cost, and
misroute cost in the answer skeleton.

- [ ] **Step 6: Mark the roadmap item complete and create the final checkpoint**

```bash
git add docs/agent-llm-optimization-interview-case.md \
  docs/treesem-interview-guide.md \
  docs/reports/agent-routing-safety-closure.md docs/roadmap.md
git commit -m "docs: explain structured agent routing evidence"
```

- [ ] **Step 7: Produce the module handoff**

Report:

```text
what changed
why taxonomy was fixed before changing models
one Rule hit call chain
one Rule miss to Open Agent call chain
tests and exact pass counts
historical metrics that remain unchanged
limitations and the condition for a future Router promotion
files containing the core code to study for interviews
```

Do not claim the new rules outperform the rejected Hybrid candidate unless a
new independent evaluation cycle later establishes that result.
