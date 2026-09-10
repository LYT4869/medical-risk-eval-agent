# Structured LLM Intent Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default keyword-led Agent routing path with a structured LLM semantic parser, strict deterministic validation and target binding, direct deterministic workflow execution, clarification, and a guarded open-agent branch.

**Architecture:** The Python Agent keeps deterministic safety before routing and C++ authorization at Tool execution. A dedicated short-prompt LLM call produces only an `IntentFrame`; trusted code extracts explicit IDs, validates the frame, binds symbolic targets, chooses a versioned recipe, and directly executes known Tools. The existing Rule/E5 implementations remain inactive historical and explicit rollback paths during migration.

**Tech Stack:** Python 3.10, Pydantic 2.13.4, asyncio, httpx 0.28.1, OpenAI-compatible function calling, FastAPI, unittest, Prometheus text metrics, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-10-structured-llm-intent-router-design.md`

## Global Constraints

- The production router reuses `TREESEM_AGENT_LLM_BASE_URL`, `TREESEM_AGENT_LLM_MODEL`, and `TREESEM_AGENT_LLM_API_KEY` with a separate short prompt.
- Router generation is fixed to `temperature=0`, `enable_thinking=false`, `max_output_tokens=384`, and required function calling.
- The Router may describe intent, symbolic target, requested aspects, constraints, unresolved references, and source evidence only.
- The Router must never output Tool names, SQL, URLs, roles, session IDs, Capability scopes, or generated business IDs.
- Exact prediction IDs and sample indexes are extracted from the current user message before the LLM request. Their source substrings are replaced by ordinal placeholders in the Router-visible message, and the frame can refer to them only by candidate index.
- Deterministic safety runs before routing; C++ resource authorization remains authoritative after target binding.
- The Router orchestrator owns retries. A single HTTP attempt performs one request; transport retry and schema repair share a maximum of two attempts.
- Initial Router limits are 3000 ms per attempt, 100 ms backoff, two attempts, and a 7000 ms total deadline.
- Startup rejects a configuration where `total_deadline_ms < per_attempt_timeout_ms * maximum_attempts + retry_backoff_ms`.
- Router infrastructure or protocol failure must not silently fall back to rules or the open agent and must execute no business Tool.
- Stable and registered composite workflows call Tools directly; only final prose generation calls the LLM with `tool_choice=none`.
- No full IntentFrame, user message, evidence substring, clinical value, business ID, credential, or Tool result may enter logs or Metrics labels.
- The default changes to `structured_llm` only after the deterministic regressions and first-stage quality gates pass.
- Existing E5 source, artifacts, and reports stay unchanged and inactive; their removal is outside this feature.
- No database migration is added for the first Router version.

---

## File and responsibility map

### New runtime files

- `agent/intent_frame.py` — strict version-one semantic frame, enums, bound target types, and dispatch result types.
- `agent/reference_extractor.py` — deterministic extraction of exact prediction IDs and explicitly labelled demo sample indexes plus Router-visible placeholder substitution.
- `agent/intent_router_prompt.py` — short Router system prompt, required function definition, and prompt SHA.
- `agent/structured_router.py` — bounded two-attempt LLM routing orchestration and protocol parsing.
- `agent/intent_validation.py` — evidence, compatibility, constraint, candidate-index, and unresolved-reference validation.
- `agent/workflow_registry.py` — versioned typed deterministic and composite recipes.
- `agent/intent_dispatch.py` — target binding and selection among workflow, clarification, guarded open agent, and failure.
- `agent/deterministic_workflow.py` — direct Tool execution, typed inter-stage argument resolution, and deterministic failure rendering.
- `agent/intent_routing_runtime.py` — `legacy_rule`, `structured_shadow`, and `structured_llm` construction and lifecycle.

### Existing runtime files changed

- `agent/llm_client.py` — configurable per-client attempt count so the Router transport performs exactly one request per orchestrated attempt.
- `agent/loop.py` — safety, route selection, direct workflow execution, finalization, clarification, and open-agent separation.
- `agent/run_guard.py` — construct allowed Tool sets from validated goals/recipes rather than keyword-derived scopes.
- `agent/workflow.py` — retain the old planner under an explicit legacy name only.
- `agent/observability.py` — Router/planner/run counters and histograms without high-cardinality labels.
- `server.py` — construct and close the new routing runtime and expose safe health metadata.

### Evaluation, deployment, and documentation files changed

- `evaluation/structured_router_cases.json` — 120-case Dev/Validation/Smoke Heldout first-stage corpus.
- `evaluation/run_structured_router_evaluation.py` — Router, planner, and end-to-end reports with separate gates.
- `evaluation/run_evaluation.py` — explicit Router mode and FakeStructuredRouter support for the existing 64 scenarios.
- `.env.example`, `docker-compose.yml`, `deploy/docker/agent.Dockerfile`, `scripts/prepare_demo.py`, `Makefile` — new runtime configuration and removal of E5 from the default image path.
- `docs/architecture.md`, `docs/m5-agent-core.md`, `docs/m9-observability-evaluation.md`, `docs/treesem-interview-guide.md`, `README.md` — final behavior, evidence, and interview explanation.

---

### Task 1: Define the strict IntentFrame and deterministic reference extraction

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/intent_frame.py`
- Create: `PythonServices/TreeSemAgent/agent/reference_extractor.py`
- Create: `PythonServices/TreeSemAgent/tests/test_intent_frame.py`
- Create: `PythonServices/TreeSemAgent/tests/test_reference_extractor.py`

**Interfaces:**
- Produces: `IntentKind`, `TargetKind`, `RequestedAspect`, `KnowledgeScope`, `UnresolvedReference`, `IntentTarget`, `IntentGoal`, `IntentConstraints`, and `IntentFrame` Pydantic models.
- Produces: `ReferenceExtraction(prediction_ids: tuple[str, ...], sample_indexes: tuple[int, ...], router_message: str)`.
- Produces: `extract_references(message: str) -> ReferenceExtraction`.
- Produces: `sanitize_router_context_text(text: str) -> str` for redacting business-ID, session-ID, JWT, Authorization, and API-key-shaped substrings in recent messages.
- Later tasks must use candidate ordinals and must not copy candidate values into the Router prompt.

- [ ] **Step 1: Write strict frame tests**

Test a minimal explanation frame, a two-goal composite frame, rejection of unknown fields, duplicate aspects, more than three goals, invalid evidence length, invalid explicit-pair shape, and non-null indexes on incompatible target kinds.

```python
frame = IntentFrame.model_validate({
    "schema_version": 1,
    "goals": [{
        "intent": "explanation",
        "target": {
            "type": "previous_prediction",
            "explicit_reference_index": None,
            "second_explicit_reference_index": None,
            "sample_reference_index": None,
        },
        "requested_aspects": ["decision_path"],
        "knowledge_scope": None,
        "evidence": ["上次的决策树"],
    }],
    "constraints": {"excluded_intents": [], "excluded_aspects": []},
    "unresolved_references": [],
    "needs_clarification": False,
})
self.assertEqual(frame.goals[0].target.type, TargetKind.PREVIOUS_PREDICTION)
```

- [ ] **Step 2: Run the frame tests and verify the missing module failure**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_frame -v
```

Expected: import failure for `agent.intent_frame`.

- [ ] **Step 3: Implement the strict frame models**

Use `ConfigDict(extra="forbid", strict=True)`. Fix the exact bounds:

```text
goals                         1..3
requested_aspects per goal    0..8, unique
evidence per goal             1..3
evidence string               1..160 characters
excluded_intents/aspects      0..8, unique
unresolved_references         0..4, unique enum values
candidate index               0..7
```

`explicit_prediction` requires exactly `explicit_reference_index`;
`explicit_prediction_pair` requires two distinct explicit indexes;
`demo_sample` requires exactly `sample_reference_index`; every other target
requires all candidate indexes to be null.

- [ ] **Step 4: Write reference extraction tests**

Cover exact lowercase IDs, multiple IDs in source order, duplicate IDs,
uppercase/short/malformed IDs, Chinese `样本索引 12`, English `sample #7`, and
ordinary numbers such as dates, probabilities, and dosage-like text that must
not become sample indexes.

```python
result = extract_references(
    "比较 pred_" + "a" * 32 + " 和 pred_" + "b" * 32)
self.assertEqual(result.prediction_ids, (
    "pred_" + "a" * 32, "pred_" + "b" * 32))
self.assertEqual(
    result.router_message,
    "比较 <prediction_ref_0> 和 <prediction_ref_1>")
```

- [ ] **Step 5: Run extraction tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_reference_extractor -v
```

Expected: import failure for `agent.reference_extractor`.

- [ ] **Step 6: Implement conservative reference extraction**

Use `pred_[0-9a-f]{32}` with token boundaries. Extract sample integers only
when immediately associated with one of `样本`, `样本索引`, `sample`, or
`sample index`; do not treat a standalone integer as a sample. Deduplicate
while preserving first occurrence and retain at most eight candidates. Replace
recognized values with `<prediction_ref_N>` and `<sample_ref_N>` in
`router_message`; redact prediction/session/JWT/API-key-shaped strings in
recent-message sanitization without making them selectable candidates.

- [ ] **Step 7: Run both test modules**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_frame \
  PythonServices.TreeSemAgent.tests.test_reference_extractor -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the contract**

```bash
git add PythonServices/TreeSemAgent/agent/intent_frame.py \
  PythonServices/TreeSemAgent/agent/reference_extractor.py \
  PythonServices/TreeSemAgent/tests/test_intent_frame.py \
  PythonServices/TreeSemAgent/tests/test_reference_extractor.py
git commit -m "feat: define structured Agent intent frames"
```

---

### Task 2: Add the short Router prompt and sole-owner retry orchestration

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/intent_router_prompt.py`
- Create: `PythonServices/TreeSemAgent/agent/structured_router.py`
- Create: `PythonServices/TreeSemAgent/tests/test_structured_router.py`
- Modify: `PythonServices/TreeSemAgent/agent/llm_client.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_llm_client.py`

**Interfaces:**
- Consumes: `IntentFrame` and `ReferenceExtraction` from Task 1.
- Produces: `RouterContext(message, recent_messages, current_prediction_available, references)`.
- Produces: `StructuredRoute(frame, usage, attempt_count, repaired)`.
- Produces: `StructuredRouterError(code)` where code is `intent_router_unavailable` or `invalid_intent_frame`.
- Produces: `StructuredIntentRouter.route(context: RouterContext) -> StructuredRoute`.

- [ ] **Step 1: Write the required-function and context-minimization tests**

Use a recording fake LLM. Assert that the call contains one function named
`route_user_request`, uses `LlmToolPolicy.required("route_user_request")`,
contains only clipped and sanitized final messages plus Boolean/count metadata,
and contains neither actual prediction IDs nor current prediction values.

```python
result = asyncio.run(router.route(RouterContext(
    message="解释 <prediction_ref_0>",
    recent_messages=(RecentMessage(role="user", content="刚才那个"),),
    current_prediction_available=True,
    references=ReferenceExtraction(
        ("pred_" + "a" * 32,), (), "解释 <prediction_ref_0>"),
)))
self.assertNotIn("pred_" + "a" * 32, json.dumps(client.requests))
self.assertEqual(result.frame.schema_version, 1)
```

- [ ] **Step 2: Write retry-budget and protocol-failure tests**

Cover first-attempt timeout then success, 429 then success, invalid frame then
one repair, transport failure followed by invalid frame with no third request,
wrong Tool name, multiple Tool calls, missing Tool call, malformed arguments,
and exhaustion before total deadline. Use a fake clock/sleeper; do not sleep in
unit tests.

- [ ] **Step 3: Run Router tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_structured_router -v
```

Expected: import failure for `agent.structured_router`.

- [ ] **Step 4: Make OpenAI-compatible retry count configurable**

Add `maximum_attempts: int = 2` to `OpenAiCompatibleConfig`, reject values
outside `1..2`, and replace the hard-coded `range(2)` with the configured
value. Preserve two attempts for the main Agent client. Router construction in
Task 7 will use `maximum_attempts=1`.

```python
@dataclass(frozen=True)
class OpenAiCompatibleConfig:
    # existing fields
    maximum_attempts: int = 2

    def __post_init__(self) -> None:
        if self.maximum_attempts not in {1, 2}:
            raise ValueError("maximum_attempts must be 1 or 2")
```

Give `LlmError` a closed safe `code` and `retryable` flag. Network failures,
429, and 5xx are retryable; other HTTP 4xx are not; malformed upstream response
envelopes use `invalid_response` and are handled as protocol failures by the
Router. Preserve the main Agent client's existing two-attempt policy while
stopping retries for non-retryable 4xx responses.

- [ ] **Step 5: Verify the HTTP client makes one Router transport attempt**

Add a test using a mocked `httpx.AsyncClient` that returns 500 twice. Assert a
client configured with `maximum_attempts=1` makes exactly one request, while
the default main Agent client still makes two.

- [ ] **Step 6: Implement the Router prompt and function definition**

The system prompt must say: describe semantics only; preserve negation and
contrast; select explicit references only by zero-based ordinal; never emit
Tool/role/permission/session/credential/ID values; use unresolved references
instead of guessing. Export `ROUTER_PROMPT_SHA256` and
`router_function_definition()`, deriving the JSON schema from `IntentFrame`.

- [ ] **Step 7: Implement the two-attempt Router orchestrator**

Attempt one sends the base prompt. A retryable transport failure may consume
attempt two. A schema/protocol failure may consume attempt two with a short
format-repair instruction. Both paths share `attempt_count < 2`; each request
uses `min(3.0, remaining_deadline)` and the whole method uses a monotonic
7-second deadline. Return safe error codes without embedding upstream text.

- [ ] **Step 8: Run Router and LLM client tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_llm_client \
  PythonServices.TreeSemAgent.tests.test_structured_router -v
```

Expected: all tests pass and no retry test performs a real sleep or network call.

- [ ] **Step 9: Commit the Router transport**

```bash
git add PythonServices/TreeSemAgent/agent/intent_router_prompt.py \
  PythonServices/TreeSemAgent/agent/structured_router.py \
  PythonServices/TreeSemAgent/agent/llm_client.py \
  PythonServices/TreeSemAgent/tests/test_structured_router.py \
  PythonServices/TreeSemAgent/tests/test_llm_client.py
git commit -m "feat: add bounded structured LLM routing"
```

---

### Task 3: Validate semantics and bind targets without trusting the LLM

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/intent_validation.py`
- Create: `PythonServices/TreeSemAgent/tests/test_intent_validation.py`

**Interfaces:**
- Consumes: `IntentFrame`, `ReferenceExtraction`, and `AgentRunRequest`.
- Produces: immutable `BoundTarget(kind, prediction_ids, sample_index)` and `BoundGoal(intent, target, requested_aspects, knowledge_scope)`.
- Produces: `ValidatedIntent(frame, goals, excluded_intents, excluded_aspects)`.
- Produces: `IntentValidationResult(validated: ValidatedIntent | None, clarification_code: str | None)`.
- Produces: `IntentFrameViolation(code)` for protocol-invalid frames; ambiguity is a clarification result, not an exception.

- [ ] **Step 1: Write evidence and compatibility tests**

Test that every evidence substring exists in the placeholder-substituted Router
message after NFKC/casefold/whitespace normalization; fabricated evidence, positive/excluded conflicts, explanation
with `general_knowledge`, comparison with one explicit ID, and knowledge with a
prediction-only aspect are rejected before Tool execution.

- [ ] **Step 2: Write trusted binding tests**

Cover:

```text
explicit ordinal 0        -> exact first source ID
two explicit ordinals     -> exact source pair
current_prediction        -> request.current_prediction.prediction_id
previous_prediction       -> symbolic, no guessed ID
latest_two_predictions    -> symbolic, no guessed IDs
demo_sample ordinal       -> exact extracted sample integer
out-of-range ordinal      -> IntentFrameViolation
missing current context   -> clarification current_prediction_missing
unresolved pronoun        -> clarification ambiguous_reference
```

- [ ] **Step 3: Run validation tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_validation -v
```

Expected: import failure for `agent.intent_validation`.

- [ ] **Step 4: Implement ordered semantic validation**

Follow the spec order: strict schema is already complete; verify candidate
indexes, evidence, excluded conflicts, Intent/Target/Aspect compatibility,
unresolved references, and then target binding. Never inspect the database or
call a Tool in this module.

- [ ] **Step 5: Implement deterministic clarification codes**

Use this finite set:

```text
sample_index_missing
current_prediction_missing
prediction_target_missing
comparison_target_missing
ambiguous_reference
conflicting_request
```

Each code maps to a fixed Chinese clarification sentence in Task 6; do not put
free-form LLM text into `clarification_code`.

- [ ] **Step 6: Run validation tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_validation -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit validation and binding**

```bash
git add PythonServices/TreeSemAgent/agent/intent_validation.py \
  PythonServices/TreeSemAgent/tests/test_intent_validation.py
git commit -m "feat: validate and bind structured Agent intents"
```

---

### Task 4: Replace keyword workflow planning with a typed recipe registry

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/workflow_registry.py`
- Create: `PythonServices/TreeSemAgent/agent/intent_dispatch.py`
- Create: `PythonServices/TreeSemAgent/tests/test_workflow_registry.py`
- Create: `PythonServices/TreeSemAgent/tests/test_intent_dispatch.py`
- Modify: `PythonServices/TreeSemAgent/agent/workflow.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_workflow.py`

**Interfaces:**
- Consumes: `ValidatedIntent` and `BoundGoal` from Task 3.
- Produces: `ArgumentSource` enum, `WorkflowStage`, `WorkflowRecipe`, and `WorkflowRegistry`.
- Produces: `DispatchKind` values `workflow`, `composite_workflow`, `clarification`, and `open_agent`.
- Produces: `DispatchDecision(kind, recipe, validated_intent, clarification_code, allowed_tools)`.
- Produces: `IntentDispatcher.dispatch(validation_result) -> DispatchDecision`.

- [ ] **Step 1: Rename the existing planner as an explicit legacy planner**

Rename `WorkflowPlanner` to `LegacyWorkflowPlanner` and add a temporary alias
`WorkflowPlanner = LegacyWorkflowPlanner` so existing tests remain green during
migration. Do not change its behavior or the E5/Rule classes.

- [ ] **Step 2: Write registry integrity tests**

Assert recipe IDs are unique, every stage uses a registered Tool, every
argument source is accepted by its Tool, recipe patterns are non-overlapping,
and `WORKFLOW_REGISTRY_VERSION` is a SHA-256 over canonical recipe metadata.

Define these version-one recipes:

```text
predict_demo_sample
read_current_or_explicit_prediction
explain_current_or_explicit_prediction
explain_previous_prediction
list_session_history
compare_latest_two
compare_explicit_pair
search_general_knowledge
activate_explanation_skill
activate_comparison_skill
activate_education_skill
compare_and_explain_latest_two
```

- [ ] **Step 3: Write dispatch behavior tests**

Test unique single-goal mapping, the registered compare-and-explain composite,
missing-target clarification, `other` to guarded open agent, an unregistered
multi-goal combination to guarded open agent, and allowed Tools equal to the
union of recipe stages with no side-effecting Tool in the open branch.

- [ ] **Step 4: Run registry and dispatch tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_workflow_registry \
  PythonServices.TreeSemAgent.tests.test_intent_dispatch -v
```

Expected: imports fail for the two new modules.

- [ ] **Step 5: Implement typed recipe metadata**

Use immutable dataclasses. Each stage contains only:

```python
@dataclass(frozen=True)
class WorkflowStage:
    tool_name: str
    argument_source: ArgumentSource
```

Argument sources are a closed enum: `BOUND_SAMPLE`, `BOUND_PREDICTION`,
`SESSION_HISTORY`, `LATEST_TWO_HISTORY`, `PREVIOUS_FROM_HISTORY`,
`PAIR_FROM_HISTORY`, `BOUND_EXPLICIT_PAIR`, `KNOWLEDGE_QUERY`, and
`TRUSTED_SKILL_ID`.

- [ ] **Step 6: Implement deterministic dispatch**

Match normalized typed fields, never message substrings. Return clarification
when Task 3 produced one. Return guarded open agent only for `other` or valid
unregistered combinations. Reject any recipe whose Tools exceed the registered
domain Tool set at startup.

- [ ] **Step 7: Run new and legacy workflow tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_workflow \
  PythonServices.TreeSemAgent.tests.test_workflow_registry \
  PythonServices.TreeSemAgent.tests.test_intent_dispatch -v
```

Expected: all tests pass; legacy tests prove rollback behavior is unchanged.

- [ ] **Step 8: Commit typed planning**

```bash
git add PythonServices/TreeSemAgent/agent/workflow.py \
  PythonServices/TreeSemAgent/agent/workflow_registry.py \
  PythonServices/TreeSemAgent/agent/intent_dispatch.py \
  PythonServices/TreeSemAgent/tests/test_workflow.py \
  PythonServices/TreeSemAgent/tests/test_workflow_registry.py \
  PythonServices/TreeSemAgent/tests/test_intent_dispatch.py
git commit -m "feat: map intent frames to typed workflows"
```

---

### Task 5: Execute deterministic recipes directly and resolve dependent arguments

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/deterministic_workflow.py`
- Create: `PythonServices/TreeSemAgent/tests/test_deterministic_workflow.py`
- Modify: `PythonServices/TreeSemAgent/agent/tool_registry.py`

**Interfaces:**
- Consumes: `WorkflowRecipe`, `ValidatedIntent`, `ToolContext`, and `ToolRegistry.execute`.
- Produces: `WorkflowExecution(tool_results, tool_usages, prediction_ids, citations, knowledge_index_version, active_skill)`.
- Produces: `DeterministicWorkflowExecutor.execute(recipe, intent, context) -> WorkflowExecution`.
- Produces: `ToolRegistry.definition_names(...) -> frozenset[str]` for startup recipe validation without exposing full schemas.

- [ ] **Step 1: Write direct-execution tests**

Use the existing fake backend and knowledge clients. Assert:

```text
predict sample     -> one direct predict_sample call
current explain   -> one direct get_explanation call with trusted current ID
previous explain  -> history, then explanation using the second history ID
latest comparison -> history, then comparison using returned IDs
explicit pair     -> comparison with exact source IDs
knowledge         -> one MCP search using current message, never patient IDs
```

Assert the LLM fake is absent from the executor constructor so these stages
cannot accidentally ask the model to choose Tool arguments.

- [ ] **Step 2: Write partial-result and Tool-error tests**

History with fewer than two predictions must return a typed
`insufficient_history` execution result without calling comparison. A failed
stage must stop later stages. Successful earlier read-only results may feed the
deterministic renderer but may not be labelled as full workflow success.

- [ ] **Step 3: Run executor tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_deterministic_workflow -v
```

Expected: import failure for `agent.deterministic_workflow`.

- [ ] **Step 4: Implement argument resolution from trusted state**

`BOUND_SAMPLE` and `BOUND_PREDICTION` come from Task 3. History-derived IDs
come only from validated `HistoryToolOutput`. `KNOWLEDGE_QUERY` uses the current
user message after removing exact prediction IDs; reject an empty query.
`TRUSTED_SKILL_ID` comes from the registry recipe constant, not Router output.

- [ ] **Step 5: Implement sequential direct Tool execution**

Call `ToolRegistry.execute` once per stage, update prediction/citation/Skill
grounding exactly as the current loop does, and stop on a failed `ToolUse`.
Preserve Tool usage ordering and attach active Skill instructions only to the
later finalization prompt.

- [ ] **Step 6: Run Tool and executor tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_deterministic_workflow \
  PythonServices.TreeSemAgent.tests.test_m7_m8_agent -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit direct workflow execution**

```bash
git add PythonServices/TreeSemAgent/agent/deterministic_workflow.py \
  PythonServices/TreeSemAgent/agent/tool_registry.py \
  PythonServices/TreeSemAgent/tests/test_deterministic_workflow.py
git commit -m "feat: execute stable Agent workflows directly"
```

---

### Task 6: Integrate structured dispatch into AgentLoop

**Files:**
- Modify: `PythonServices/TreeSemAgent/agent/loop.py`
- Modify: `PythonServices/TreeSemAgent/agent/run_guard.py`
- Modify: `PythonServices/TreeSemAgent/agent/prompt.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_agent_loop.py`
- Create: `PythonServices/TreeSemAgent/tests/test_structured_agent_loop.py`

**Interfaces:**
- Consumes: `StructuredIntentRouter`, `IntentDispatcher`, and `DeterministicWorkflowExecutor`.
- Produces: `AgentLoop(..., routing_runtime: IntentRoutingRuntime)` behavior for legacy, shadow, and structured modes.
- Produces: `AgentRunGuard.for_allowed_tools(allowed_tools: set[str]) -> AgentRunGuard` for the structured open-agent branch.
- Preserves: `AgentRunResponse` public schema and existing grounding policy.

- [ ] **Step 1: Extract the current loop into explicit legacy/open/finalize methods**

Before changing behavior, move current keyword-plan execution to
`_run_legacy_steps`, open-agent execution to `_run_open_agent`, and final answer
validation/repair to `_finalize_answer`. Run the current Agent tests after this
mechanical refactor to prove behavior is unchanged.

- [ ] **Step 2: Write the browser regression as a structured-loop test**

For a request with `current_prediction` and message `解释一下刚刚的结果`, feed
an explanation IntentFrame and one final-answer LLM turn. Assert the backend
receives exactly one `get_explanation` call, the LLM is called only for the
Router and final prose, no LLM Tool call is requested, and the response contains
the trusted prediction ID.

- [ ] **Step 3: Write negation, temporal, composite, and clarification tests**

Cover the eight acceptance examples from the spec. In particular:

```text
不要解释结果，看看上次的决策树
```

must treat the negation as excluding a general result summary while preserving
the positive request for the previous decision path, then use history plus the
previous record selected by the validated positive goal. `解释那个` with
no resolvable target must return a fixed clarification with zero Tool calls.

- [ ] **Step 4: Write Router failure safety tests**

Assert `intent_router_unavailable` and `invalid_intent_frame` raise safe
`AgentExecutionError` codes, invoke neither legacy rules nor open agent, and
produce zero backend/MCP calls. Add both codes to `EXECUTION_ERROR_CODES`.

- [ ] **Step 5: Write guarded open-agent tests**

Pass a valid `other` frame and assert the model sees no side-effecting
`predict_sample`, feedback, Admin, or arbitrary network Tool. Pass a read-only
unregistered explanation/comparison combination and assert only the union of
validated read-only Tools is exposed.

- [ ] **Step 6: Run structured-loop tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_structured_agent_loop -v
```

Expected: failures because `AgentLoop` has not consumed structured dispatch.

- [ ] **Step 7: Implement the structured branch**

The order inside `_run_steps` becomes safety, routing, dispatch, and branch
execution. For workflows, execute Task 5 directly and make one final LLM call
with no Tool definitions. For clarification, return the fixed message without
LLM or Tool calls. For open agent, derive `AgentRunGuard` from
`DispatchDecision.allowed_tools`. Keep citation repair and response grounding.

- [ ] **Step 8: Add deterministic finalization fallback**

If final prose generation fails after successful workflow execution, return a
minimal Chinese response selected by recipe renderer kind:

```text
prediction_completed
prediction_data_available
explanation_data_available
history_data_available
comparison_data_available
knowledge_temporarily_unrendered
```

Include only already validated grounding IDs and no invented probability or
clinical fact.

- [ ] **Step 9: Run the full Agent runtime unit suite**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest discover \
  -s PythonServices/TreeSemAgent/tests -p 'test_*.py' -v
```

Expected: all tests pass; the old loop tests use the explicit legacy test
runtime until Task 7 changes the production default.

- [ ] **Step 10: Commit AgentLoop integration**

```bash
git add PythonServices/TreeSemAgent/agent/loop.py \
  PythonServices/TreeSemAgent/agent/run_guard.py \
  PythonServices/TreeSemAgent/agent/prompt.py \
  PythonServices/TreeSemAgent/tests/test_agent_loop.py \
  PythonServices/TreeSemAgent/tests/test_structured_agent_loop.py
git commit -m "feat: dispatch Agent requests through structured intents"
```

---

### Task 7: Build runtime modes, configuration, health metadata, and lifecycle

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/intent_routing_runtime.py`
- Create: `PythonServices/TreeSemAgent/tests/test_intent_routing_runtime.py`
- Modify: `PythonServices/TreeSemAgent/server.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_server_routing.py`

**Interfaces:**
- Produces: `IntentRoutingMode` with `LEGACY_RULE`, `STRUCTURED_SHADOW`, and `STRUCTURED_LLM`.
- Produces: `IntentRoutingSettings.from_environment()`.
- Produces: `IntentRoutingRuntime.route(request)`, metadata properties, and `close()`.
- Consumes one main Agent LLM and owns a separate Router `OpenAiCompatibleClient(maximum_attempts=1)` in real mode.

- [ ] **Step 1: Write exact environment validation tests**

Parse these variables:

```text
TREESEM_AGENT_ROUTING_MODE=legacy_rule|structured_shadow|structured_llm
TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS=3000
TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS=7000
TREESEM_AGENT_ROUTER_MAX_ATTEMPTS=2
TREESEM_AGENT_ROUTER_RETRY_BACKOFF_MS=100
TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES=4
TREESEM_AGENT_ROUTER_CONTEXT_MAX_CHARS=6000
```

Reject nonpositive limits, more than two attempts, and impossible deadline
budgets. `structured_llm` with real LLM mode requires the same base URL and
model already required by the Agent.

- [ ] **Step 2: Write mode behavior tests**

`legacy_rule` executes only old rules. `structured_shadow` calls the structured
Router and validates/plans it but returns the legacy execution decision;
structured failure is recorded and does not alter the legacy decision.
`structured_llm` is authoritative and never calls legacy routing.

- [ ] **Step 3: Write scripted-demo and lifecycle tests**

Provide a deterministic `FakeStructuredRouter` for local offline demo/tests.
Assert the Router client is closed exactly once, the main Agent client is still
closed exactly once, and injected-loop health tests construct neither client.

- [ ] **Step 4: Run runtime/server tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_routing_runtime \
  PythonServices.TreeSemAgent.tests.test_server_routing -v
```

Expected: failures for missing structured runtime types and metadata.

- [ ] **Step 5: Implement runtime construction without touching E5 modules**

Keep `routing_config.py`, `semantic_routing.py`, artifacts, and their tests as
historical modules. `server.py` imports only `intent_routing_runtime.py` for the
production path. Build the Router client with the current base URL/model/key,
fixed zero temperature/thinking disabled/384 tokens, and one transport attempt.

- [ ] **Step 6: Expose safe health/readiness metadata**

Return:

```json
{
  "routing_mode": "structured_llm",
  "router_model": "configured-model-name",
  "router_prompt_sha256": "64 lowercase hex",
  "intent_frame_schema_version": 1,
  "workflow_registry_version": "64 lowercase hex"
}
```

Do not call the cloud model from `/health` or `/ready`. Do not expose base URL,
API key, prompts, thresholds, user messages, or frames.

- [ ] **Step 7: Run runtime/server and full Agent tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_intent_routing_runtime \
  PythonServices.TreeSemAgent.tests.test_server_routing -v
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest discover \
  -s PythonServices/TreeSemAgent/tests -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 8: Commit runtime modes**

```bash
git add PythonServices/TreeSemAgent/agent/intent_routing_runtime.py \
  PythonServices/TreeSemAgent/server.py \
  PythonServices/TreeSemAgent/tests/test_intent_routing_runtime.py \
  PythonServices/TreeSemAgent/tests/test_server_routing.py
git commit -m "feat: add structured routing runtime modes"
```

---

### Task 8: Add low-cardinality Router, planner, and end-to-end observability

**Files:**
- Modify: `PythonServices/TreeSemAgent/agent/observability.py`
- Modify: `PythonServices/TreeSemAgent/agent/structured_router.py`
- Modify: `PythonServices/TreeSemAgent/agent/intent_routing_runtime.py`
- Modify: `PythonServices/TreeSemAgent/agent/loop.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_observability.py`
- Create: `PythonServices/TreeSemAgent/tests/test_routing_observability.py`

**Interfaces:**
- Produces fixed-label counters/histograms for Router result, dispatch kind, attempts, repairs, latency, per-run LLM calls/tokens/latency, and end-to-end result.
- Produces one safe `agent.intent_router` Trace span and one `agent.intent_dispatch` span per structured request.

- [ ] **Step 1: Write Metrics label-safety tests**

Render Metrics after requests containing prediction IDs and unique user text.
Assert the output contains only enum labels such as `result`, `dispatch`, and
`attempts`, and contains no request text, evidence, prediction/session/request
ID, model response, or Tool argument.

- [ ] **Step 2: Write accounting tests**

For a one-attempt workflow with one finalization call, assert:

```text
router attempts = 1
LLM calls per run = 2
workflow rate increments once
open-agent rate does not increment
total tokens per run = router usage + finalization usage
```

For one repair plus finalization, expect three total LLM calls. For a safety
refusal, expect zero Router and zero LLM calls.

- [ ] **Step 3: Run observability tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_observability \
  PythonServices.TreeSemAgent.tests.test_routing_observability -v
```

Expected: missing Router/planner/run series assertions fail.

- [ ] **Step 4: Implement fixed series**

Use these names and closed labels:

```text
treesem_agent_intent_router_requests_total{result}
treesem_agent_intent_router_repairs_total{result}
treesem_agent_intent_router_duration_seconds{result}
treesem_agent_intent_dispatch_total{dispatch}
treesem_agent_workflow_executions_total{result,recipe_class}
treesem_agent_llm_calls_per_run
treesem_agent_llm_tokens_per_run{kind}
treesem_agent_llm_duration_per_run_seconds
```

Use `recipe_class=single|composite`; never label with individual recipe IDs.

- [ ] **Step 5: Implement safe Trace events**

Trace fields may include Router result, attempt count, repaired Boolean,
dispatch kind, goal count, and duration. Do not pass `IntentFrame.model_dump()`
or evidence. Mark infrastructure/protocol failures with stable error codes.

- [ ] **Step 6: Run observability and full Agent tests**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_observability \
  PythonServices.TreeSemAgent.tests.test_routing_observability -v
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest discover \
  -s PythonServices/TreeSemAgent/tests -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 7: Commit observability**

```bash
git add PythonServices/TreeSemAgent/agent/observability.py \
  PythonServices/TreeSemAgent/agent/structured_router.py \
  PythonServices/TreeSemAgent/agent/intent_routing_runtime.py \
  PythonServices/TreeSemAgent/agent/loop.py \
  PythonServices/TreeSemAgent/tests/test_observability.py \
  PythonServices/TreeSemAgent/tests/test_routing_observability.py
git commit -m "feat: observe structured Agent routing"
```

---

### Task 9: Add the 120-case first-stage evaluation and preserve the 64-case gate

**Files:**
- Create: `PythonServices/TreeSemAgent/evaluation/structured_router_cases.json`
- Create: `PythonServices/TreeSemAgent/evaluation/run_structured_router_evaluation.py`
- Create: `PythonServices/TreeSemAgent/tests/test_structured_router_evaluation.py`
- Modify: `PythonServices/TreeSemAgent/evaluation/run_evaluation.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_evaluation.py`
- Modify: `Makefile`

**Interfaces:**
- Produces: `make intent-router-evaluate` for Router/planner evaluation.
- Produces: `make intent-router-regression` for the existing deterministic Agent suite with `FakeStructuredRouter`.
- Produces JSON reports with `router`, `planner`, and `end_to_end` top-level sections.

- [ ] **Step 1: Define and test the evaluation-case schema**

Each case contains:

```json
{
  "case_id": "sr_dev_001",
  "split": "dev",
  "slice": ["explanation", "current_reference"],
  "actor_role": "patient",
  "message": "解释一下刚刚的结果",
  "recent_messages": [],
  "current_prediction_available": true,
  "expected_frame": {
    "intents": ["explanation"],
    "target_types": ["current_prediction"],
    "required_aspects": [],
    "excluded_aspects": []
  },
  "expected_dispatch": "workflow",
  "expected_recipe": "explain_current_or_explicit_prediction",
  "critical": false
}
```

Tests enforce exactly 60 Dev, 30 Validation, 30 Smoke Heldout; unique IDs; no
duplicate normalized messages across splits; every critical slice present in
Validation and Heldout; and no concrete patient data.

- [ ] **Step 2: Author the first-stage corpus**

Distribute the 120 cases across simple six-domain intents, skill requests,
`other`, negation, contrast, current/previous/latest references, explicit IDs,
sample indexes, registered composites, unregistered combinations,
clarification, Chinese/English colloquial wording, prompt injection, fabricated
IDs, cross-patient requests, and privilege escalation. Mark security, generated
ID, unauthorized Tool, and urgent-medical cases critical.

- [ ] **Step 3: Write scoring tests before the runner**

Use a Fake Router with one intentionally wrong target. Assert Router metrics
count the target error without incorrectly lowering schema validity; use a
planner mistake to lower workflow mapping only; use a blocked Tool attempt to
affect only end-to-end safety.

- [ ] **Step 4: Run evaluation tests and verify they fail**

Run:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest \
  PythonServices.TreeSemAgent.tests.test_structured_router_evaluation -v
```

Expected: missing runner/schema behavior fails.

- [ ] **Step 5: Implement separate layer scoring**

Router section reports schema validity, intent, target, aspect, constraint, and
hallucinated-ID count. Planner reports workflow/composite/clarification/open
accuracy. End-to-end reports task success, unauthorized Tool execution,
grounding, calls, tokens, LLM latency, and total latency. A wrong layer must not
be double-counted as a different layer's primary metric.

- [ ] **Step 6: Add deterministic and real execution modes**

`--mode fake` uses stored frames and no network. `--mode real` uses the current
OpenAI-compatible Router client, writes model/prompt/schema/registry versions,
and returns `status=not_run` when credentials are absent. `--split` accepts
`dev`, `validation`, or `smoke_heldout`; do not expose final Heldout cases to
prompt tuning.

- [ ] **Step 7: Adapt the existing 64-scenario Agent evaluator**

Inject `FakeStructuredRouter` frames derived from each existing fixture's
declared expected tools. Do not use keywords to decide expected behavior during
the test. Preserve the current hard gates for grounding, citations, medical
boundary, and security.

- [ ] **Step 8: Add Make targets and run offline gates**

Add:

```make
intent-router-evaluate:
	PYTHONPATH=$(ROUTING_ROOT) python3 $(ROUTING_ROOT)/evaluation/run_structured_router_evaluation.py --mode fake --split dev

intent-router-regression:
	PYTHONPATH=$(ROUTING_ROOT) python3 $(ROUTING_ROOT)/evaluation/run_evaluation.py --mode deterministic --routing-mode structured_llm
```

Run both targets and the full Python suite. Expected: 120-case schema checks,
fake layer scoring, and all 64 deterministic scenarios pass.

- [ ] **Step 9: Commit the evaluation framework**

```bash
git add PythonServices/TreeSemAgent/evaluation/structured_router_cases.json \
  PythonServices/TreeSemAgent/evaluation/run_structured_router_evaluation.py \
  PythonServices/TreeSemAgent/evaluation/run_evaluation.py \
  PythonServices/TreeSemAgent/tests/test_structured_router_evaluation.py \
  PythonServices/TreeSemAgent/tests/test_evaluation.py Makefile
git commit -m "test: evaluate structured Agent routing"
```

---

### Task 10: Run real-model validation and promote only when gates pass

**Files:**
- Create: `docs/reports/structured-intent-router-evaluation.md`
- Create locally, ignored: `artifacts/evaluation/structured-router/*.json`
- Modify only if Dev/Validation analysis requires it: `agent/intent_router_prompt.py`, `agent/intent_validation.py`, `agent/workflow_registry.py`
- Modify corresponding tests for every accepted Dev/Validation correction.

**Interfaces:**
- Consumes the current local OpenAI-compatible credentials without printing or committing them.
- Produces measured warmed p50/p95/p99, quality rates, total tokens per run, and promotion decision.

- [ ] **Step 1: Run Dev with the real configured model**

Run:

```bash
set -a
source .env
set +a
PYTHONPATH=PythonServices/TreeSemAgent python3 \
  PythonServices/TreeSemAgent/evaluation/run_structured_router_evaluation.py \
  --mode real --split dev \
  --output artifacts/evaluation/structured-router/dev.json
```

Do not echo environment variables. Record request count and token cost before
running Validation.

- [ ] **Step 2: Fix only reproducible Dev failures with tests first**

Classify failures as prompt, schema, validator, recipe, infrastructure, or
evaluation-label defects. Add a failing unit case before each code change. Do
not alter expected behavior merely to match model output and do not inspect
Smoke Heldout.

- [ ] **Step 3: Run Validation once per candidate revision**

Run the same command with `--split validation`. Freeze prompt SHA, schema
version, and workflow registry version after meeting:

```text
schema valid rate         >= 99%
intent accuracy           >= 90%
target accuracy           >= 95%
constraint accuracy       >= 90%
clarification accuracy    >= 90%
workflow mapping accuracy >= 90%
hallucinated IDs          = 0
unauthorized execution    = 0
critical safety           = 100%
```

- [ ] **Step 4: Reveal Smoke Heldout exactly once**

Run `--split smoke_heldout`, archive the JSON with Prompt/Schema/Registry/model
metadata, and make no tuning change based on this result. If a hard safety gate
fails, keep `legacy_rule` as default and record the non-promotion decision.

- [ ] **Step 5: Benchmark Router latency and total per-run cost**

Warm with 20 non-scored requests, then measure at least 100 Router calls.
Report Router p50/p95/p99 and end-to-end LLM calls, total tokens, total LLM
latency, workflow/open-agent rates, and task success. Do not claim Router-token
reduction as whole-run cost reduction.

- [ ] **Step 6: Write the evidence report**

The report must state model ID/date, case counts, immutable hashes, every gate,
failure categories, comparison against `legacy_rule`, comparison against the
historical E5 experiment, token/cost calculation, and explicit promotion or
rollback decision. Mark the future 200–300 unseen final Heldout as a separate
promotion-strengthening activity, not completed evidence.

- [ ] **Step 7: Commit measured evidence and any tested corrections**

```bash
git add PythonServices/TreeSemAgent/agent/intent_router_prompt.py \
  PythonServices/TreeSemAgent/agent/intent_validation.py \
  PythonServices/TreeSemAgent/agent/workflow_registry.py \
  PythonServices/TreeSemAgent/tests/test_structured_router.py \
  PythonServices/TreeSemAgent/tests/test_intent_validation.py \
  PythonServices/TreeSemAgent/tests/test_workflow_registry.py \
  docs/reports/structured-intent-router-evaluation.md
git commit -m "test: validate structured routing with a real model"
```

If there were no runtime corrections, stage only the report. Never add `.env`
or raw local evaluation payloads containing prompts.

---

### Task 11: Update deployment defaults, demo, and project documentation

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `deploy/docker/agent.Dockerfile`
- Modify: `scripts/prepare_demo.py`
- Modify: `PythonServices/TreeSemAgent/tests/test_routing_packaging.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/m5-agent-core.md`
- Modify: `docs/m9-observability-evaluation.md`
- Modify: `docs/treesem-interview-guide.md`

**Interfaces:**
- Produces a real/demo deployment whose configured default follows Task 10's promotion decision.
- Preserves explicit `legacy_rule` rollback and leaves E5 source/history available outside the runtime image path.

- [ ] **Step 1: Write packaging failures first**

Update packaging tests to require the new mode and timeout variables, reject an
impossible Router deadline, assert no default E5 artifact mount, and assert the
offline scripted demo uses a FakeStructuredRouter without a cloud call.

- [ ] **Step 2: Update environment and Compose configuration**

Replace the default E5 variables in the Agent service with the Router timeout,
attempt, context, and mode variables. Remove
`TREESEM_INSTALL_SEMANTIC_ROUTING` and `/routing/artifact` from the default Agent
image/service. Keep standalone historical E5 export/benchmark targets unchanged.

If Task 10 passes all hard gates, set:

```text
TREESEM_AGENT_ROUTING_MODE=structured_llm
```

Otherwise use `legacy_rule` and document the failed promotion gate.

- [ ] **Step 3: Update prepare-demo validation**

Validate the three new modes and the timeout equation. `scripted_demo` must not
require cloud credentials. `demo-real` with structured routing requires base
URL and model; the API key may be empty only for a local unauthenticated
OpenAI-compatible endpoint.

- [ ] **Step 4: Rebuild and run the browser acceptance flow**

Run:

```bash
docker compose build agent backend demo-web
docker compose up -d mysql model-adapter knowledge backend agent demo-web
python3 scripts/demo_flow.py
```

Then submit `解释一下刚刚的结果` through the Web UI/API and confirm a 200
response, one explanation Tool call, valid grounding, and no
`tool_not_allowed`. Also verify negation, previous result, latest comparison,
knowledge citation, and clarification.

- [ ] **Step 5: Exercise failure and rollback behavior**

Point the Router at an unavailable endpoint and verify no Tool executes and the
Agent returns a safe 502. Restart with `legacy_rule` and verify the old path is
restored only through configuration. Confirm `/health` stays responsive during
a 3-second Router timeout.

- [ ] **Step 6: Run full verification**

Run:

```bash
git diff --check
PYTHONPATH=PythonServices/TreeSemAgent python3 -m unittest discover \
  -s PythonServices/TreeSemAgent/tests -p 'test_*.py' -v
make intent-router-regression
python3 scripts/verify_full.py --with-compose
```

Expected: all unit, deterministic evaluation, security, container, and browser
checks pass. Record any artifact-dependent test explicitly skipped by the
verification script; do not describe a skipped test as passed.

- [ ] **Step 7: Update architecture and interview documentation**

Explain the four boundaries in interview language:

```text
LLM performs constrained semantic parsing
trusted code validates and binds targets
registered workflows execute known Tools directly
C++ enforces ownership and authorization
```

Include the measured routing quality, total token/call/latency comparison,
negation/browser failure case, retry-budget rationale, explicit-ID anti-
hallucination design, and why E5 remains historical rather than default.

- [ ] **Step 8: Commit deployment and documentation**

```bash
git add .env.example docker-compose.yml deploy/docker/agent.Dockerfile \
  scripts/prepare_demo.py PythonServices/TreeSemAgent/tests/test_routing_packaging.py \
  README.md docs/architecture.md docs/m5-agent-core.md \
  docs/m9-observability-evaluation.md docs/treesem-interview-guide.md
git commit -m "feat: deploy structured Agent intent routing"
```

---

## Final verification checklist

- [ ] Default normal requests use `structured_llm` only if the measured hard gates passed.
- [ ] Safety refusal paths make zero Router, LLM, and Tool calls.
- [ ] Router output cannot contain executable Tool selection or generated business IDs.
- [ ] Explicit IDs are hidden from the Router-visible message and accepted only when copied through a valid source-candidate ordinal.
- [ ] Stable and composite workflows perform no LLM Tool-planning turns.
- [ ] Clarification performs zero Tool calls and is not reported as infrastructure failure.
- [ ] Router timeout, 429, 5xx, and invalid schema consume at most two total attempts.
- [ ] Router failure never automatically enters legacy rules or the open agent.
- [ ] Guarded open agent has no prediction-creation or administrative side-effect Tool.
- [ ] Current, previous, latest-two, explicit-ID, negation, contrast, and composite targets are covered.
- [ ] Existing 64 deterministic Agent scenarios pass through `FakeStructuredRouter`.
- [ ] First-stage 120-case evaluation reports Router, planner, and end-to-end metrics separately.
- [ ] Metrics and Trace contain no user text, evidence, IDs, credentials, clinical values, or Tool payloads.
- [ ] Health exposes only mode/model and immutable version hashes and makes no cloud request.
- [ ] Browser regression no longer returns `tool_not_allowed` for “解释一下刚刚的结果”.
- [ ] Explicit `legacy_rule` restart rollback works; request-level silent fallback does not exist.
- [ ] E5 code and historical reports remain present but inactive in the default runtime and image.
- [ ] No `.env`, API key, raw live-model payload, or patient-derived data is committed.
- [ ] `git diff --check`, full Python tests, deterministic evaluation, and full Compose verification pass.
