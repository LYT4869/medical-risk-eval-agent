# Structured LLM Intent Router Design

## 1. Purpose

Replace the default business-intent routing path with an LLM-based structured
semantic parser. Every normal chat request passes through the router after
request-level safety checks. The router describes user intent, targets,
requested aspects, and negative constraints; deterministic code validates that
description and decides whether to execute a stable workflow, a registered
composite workflow, a clarification response, or the guarded open agent.

The router never decides authorization, selects executable Tool names, or
creates business identifiers. C++ remains the business truth source and the
final enforcement point for Capability, RBAC, session ownership, and resource
access.

This change does not reintroduce the E5 semantic router. Existing E5 code and
reports remain only as historical experiment evidence and are not part of the
new default runtime path.

## 2. Goals

- Understand negation, contrast, temporal references, requested output aspects,
  and multiple user goals without expanding a global keyword list indefinitely.
- Separate semantic understanding from workflow selection and Tool execution.
- Bind symbolic targets such as `current_prediction` and
  `previous_prediction` to trusted session state in deterministic code.
- Execute registered workflows without asking the LLM to emit already-known
  Tool calls.
- Preserve guarded open-agent behavior for valid requests that do not map to a
  stable workflow.
- Fail safely when the router infrastructure or protocol fails.
- Measure router quality separately from planner quality and end-to-end Agent
  outcomes.

## 3. Non-goals

- The router does not diagnose, prescribe, or make clinical decisions.
- The router does not replace request safety checks, C++ authorization,
  Capability validation, Tool schemas, or response grounding.
- The router does not receive clinical feature vectors, prediction values,
  complete Tool results, credentials, or database access.
- The router does not output Tool names, SQL, URLs, user roles, Session IDs,
  Capability scopes, or arbitrary execution plans.
- This change does not build a general workflow language, dynamic plugin
  system, or multi-agent planner.
- This change does not tune, promote, or remove the historical E5 artifacts.

## 4. End-to-end architecture

```text
User request
  -> C++ authentication, session resolution, and request-level checks
  -> Python deterministic safety precheck
  -> deterministic explicit-reference extraction
  -> Structured LLM Router
  -> strict IntentFrame schema validation
  -> semantic consistency validation
  -> symbolic target binding
  -> resource-level authorization at Tool execution
  -> deterministic dispatch
       -> stable workflow
       -> registered composite workflow
       -> clarification
       -> guarded open agent
       -> safe refusal/failure
  -> Tool execution
  -> final answer generation or deterministic fallback rendering
  -> prediction/citation grounding and output-policy validation
```

Request-level safety runs before the router so explicit credential attacks,
privilege escalation, urgent symptoms, and prohibited individualized treatment
requests do not consume an LLM call. Resource-level authorization runs after
target binding because the protected resource is not known earlier. Safety and
authorization therefore remain layered rather than being delegated to the
router.

## 5. Model and protocol

The router reuses the currently configured OpenAI-compatible model, base URL,
and API key. It has an independent short system prompt and independent runtime
limits:

```text
temperature          = 0
enable_thinking      = false
max_output_tokens    = 384
parallel_tool_calls  = false
```

The model receives one required, side-effect-free function definition named
`route_user_request`. Its arguments are the IntentFrame. The function is never
registered with the business Tool registry and calling it cannot access C++,
MySQL, ONNX, MCP, or patient resources.

Required function calling is preferred over unconstrained JSON text because the
existing OpenAI-compatible client already supports required Tool selection.
The returned arguments remain untrusted until strict schema and semantic
validation finish.

## 6. Router context

The router receives only:

- the current user message after deterministically extracted business references
  have been replaced by ordinal placeholders;
- at most the four most recent persisted final user/assistant messages, clipped
  to a fixed total character budget;
- `current_prediction_available: bool`;
- the count and ordinal names of deterministically extracted explicit
  prediction-ID candidates;
- the count and ordinal names of deterministically extracted sample-index
  candidates.

The router does not directly access or receive structured clinical database
records. Recent messages are database-derived conversation context, but they are
cropped final messages rather than raw clinical records or Tool traces.

The router does not receive the Agent main prompt. It receives its own short
router system prompt and the IntentFrame function schema. It does not receive
JWTs, cookies, capability tokens, patient identifiers, prediction labels,
probabilities, important-feature values, complete explanations, Tool arguments,
or Tool results.

## 7. Explicit reference extraction

Before the LLM call, deterministic code extracts exact business references from
the current user message:

```text
prediction ID: pred_<32 lowercase hexadecimal characters>
sample index:  non-negative integer in a supported sample expression
```

The actual values are stored outside the LLM frame. Their occurrences in the
Router-visible message are replaced by `<prediction_ref_N>` and
`<sample_ref_N>` placeholders. The router can select only an ordinal candidate:

```json
{
  "type": "explicit_prediction",
  "explicit_reference_index": 0
}
```

The validator maps that index back to the exact source value. An out-of-range
index is an invalid frame. This prevents the model from generating a business
ID that was not present in the user message.

Symbolic references such as `current_prediction`, `previous_prediction`, and
`latest_two_predictions` never contain IDs. They are resolved through trusted
session state and history Tools.

## 8. IntentFrame schema

The strict version-one frame is conceptually:

```json
{
  "schema_version": 1,
  "goals": [
    {
      "intent": "explanation",
      "target": {
        "type": "previous_prediction",
        "explicit_reference_index": null,
        "sample_reference_index": null
      },
      "requested_aspects": ["decision_path"],
      "knowledge_scope": null,
      "evidence": ["上次的决策树"]
    }
  ],
  "constraints": {
    "excluded_intents": [],
    "excluded_aspects": ["prediction_summary"]
  },
  "unresolved_references": [],
  "needs_clarification": false
}
```

All objects forbid unknown fields. Lists have fixed maximum lengths, strings
have fixed length limits, and duplicate enum values are rejected.

### 8.1 Intent values

```text
prediction
summary
explanation
history
comparison
knowledge
skill
other
```

`other` means the request is valid but has no registered deterministic business
workflow. It is a candidate for the guarded open agent, not an error.

### 8.2 Target values

```text
demo_sample
current_prediction
previous_prediction
latest_two_predictions
explicit_prediction
explicit_prediction_pair
session_history
general_knowledge
none
```

### 8.3 Requested aspect values

```text
prediction_summary
label
probability
confidence
model_version
important_features
decision_path
history_items
comparison_changes
knowledge_overview
citations
```

### 8.4 Constraints

`excluded_intents` and `excluded_aspects` preserve negated or contrasted user
requirements. A positive goal that is also excluded is a semantic conflict and
cannot enter a stable workflow.

### 8.5 Evidence

Each goal includes one or more short, exact substrings from the placeholder-
substituted Router-visible message. Evidence is checked against that normalized
text. Evidence helps detect fabricated goals but is not treated as proof of
correctness. Evidence text is not persisted or emitted in logs.

The schema deliberately excludes model-reported confidence because an
uncalibrated confidence value must not control execution.

## 9. Validation and target binding

Validation is deterministic and ordered:

1. Validate the required function call and strict Pydantic schema.
2. Validate list sizes, uniqueness, enum compatibility, and candidate indexes.
3. Verify every evidence string exists in the current message.
4. Reject goals that conflict with excluded intents or aspects.
5. Verify Intent, Target, and requested-aspect compatibility.
6. Detect missing target information and unresolved references.
7. Convert explicit candidate indexes to the exact deterministically extracted
   values.
8. Preserve symbolic targets for workflow planning.
9. Let the C++ Tool endpoint enforce resource ownership and RBAC when a real ID
   is used.

Target precedence is:

```text
user-supplied explicit prediction reference
  > explicit previous/latest-two temporal reference
  > explicit current-result reference
  > trusted current-prediction context used only to complete an otherwise
    unambiguous current-result request
```

Session context may complete a missing current target but never overrides an
explicit `previous_prediction`, `latest_two_predictions`, or explicit-ID target.

## 10. Deterministic dispatch outcomes

The validator and planner, not the LLM, choose one of five outcomes.

### 10.1 Stable workflow

A single complete goal maps uniquely to a registered recipe.

### 10.2 Registered composite workflow

Multiple goals map to a finite, explicitly registered composition. The first
version supports only combinations needed by the existing project, such as
comparison followed by decision-path explanations for the same two records.
Unknown combinations are not dynamically synthesized.

### 10.3 Clarification

Missing or conflicting targets produce a deterministic clarification response.
Clarification does not call a Tool or another LLM. Examples include a missing
sample index, an unresolved "that result", or a comparison with fewer than two
identifiable records.

### 10.4 Guarded open agent

`other` requests and valid goal combinations without a stable recipe enter the
open agent. The allowed Tool set is derived from validated goals and remains
bounded by Capability and RBAC. Side-effecting prediction creation is not
available to the open agent.

### 10.5 Safe refusal or failure

Safety denials, router infrastructure failures, unrepaired protocol failures,
and authorization denials do not execute a business Tool.

## 11. Workflow registry and execution

The versioned workflow registry maps Intent, Target, and Aspect combinations to
typed recipes. Initial mappings are:

| Intent frame | Deterministic recipe |
|---|---|
| prediction + demo sample + sample index | `predict_sample` |
| summary + current/explicit prediction | `get_prediction` |
| explanation + current/explicit prediction | `get_explanation` |
| explanation + previous prediction | `get_prediction_history(limit=2)` then `get_explanation(previous_id)` |
| history + session history | `get_prediction_history` |
| comparison + latest two | `get_prediction_history(limit=2)` then `compare_predictions` |
| comparison + explicit pair | `compare_predictions` |
| knowledge + general knowledge | `search_medical_knowledge` |
| explicit trusted Skill request | the registered Skill recipe |

Each recipe declares its accepted frame pattern, typed stages, parameter
sources, side-effect classification, allowed actor categories for early
rejection, and deterministic fallback-renderer type. Actor declarations are not
authorization truth; C++ remains authoritative.

The workflow executor invokes registered Tools directly. It resolves later-step
arguments from earlier typed Tool results. It does not ask the LLM to emit a
known Tool call. After execution, the final-answer model receives cropped Tool
evidence with `tool_choice=none`.

If final answer generation fails after a stable workflow completed, a
deterministic renderer returns a safe minimal summary and states that natural
language explanation is temporarily unavailable. A successful prediction or
query is not discarded merely because final prose generation failed.

## 12. Guarded open-agent policy

The open agent receives the validated, sanitized frame and a narrowed Tool set:

- stored-prediction goals expose only required read-only prediction Tools;
- knowledge goals expose only medical knowledge retrieval;
- comparison/explanation compositions expose only history, comparison, and
  explanation Tools;
- `other` defaults to no side-effecting Tools.

The open agent cannot autonomously create predictions, submit physician
feedback, modify users or assignments, expand Capability, access another
patient, or create arbitrary network clients. Existing step, Tool-call,
repetition, total-deadline, grounding, and response-policy controls remain.

## 13. Timeout and retry ownership

The router orchestrator is the only retry owner. The underlying HTTP attempt
performs exactly one request. This avoids multiplying the existing client retry
loop with an outer router retry.

The initial bounded configuration is:

```text
per-attempt request timeout = 3000 ms
retry backoff              = 100 ms
maximum attempts           = 2
total router deadline      = 7000 ms
```

Startup validation requires:

```text
total_deadline_ms >=
    per_attempt_timeout_ms * maximum_attempts + retry_backoff_ms
```

Network errors, HTTP 429, HTTP 5xx, and timeouts may consume the second attempt.
Schema-invalid output may receive at most one format-only repair within the same
total deadline. Transport retry and format repair share the two-attempt budget;
the router never performs both additional retries.

Before finalizing production defaults, the short router prompt is benchmarked
and warmed p50/p95/p99 are recorded. Configuration may then be adjusted while
preserving the startup budget invariant.

## 14. Failure semantics

### 14.1 Infrastructure failure

Network, 429, 5xx, or timeout exhaustion returns
`intent_router_unavailable`. It does not fall back silently to legacy rules or
the open agent and does not execute a business Tool.

### 14.2 Protocol failure

Missing required function call, invalid JSON arguments, unknown fields, invalid
enums, and type/length violations receive at most one format-only repair. A
second failure returns `invalid_intent_frame` without Tool execution.

### 14.3 Valid but uncertain semantics

A schema-valid frame with unresolved references, incompatible goals, or no
unique recipe is a normal semantic outcome. It produces clarification or the
guarded open agent rather than an infrastructure error.

## 15. Observability

Trace and Metrics use low-cardinality, non-sensitive values. They do not contain
the user message, evidence strings, complete IntentFrame, business IDs,
clinical values, or Tool results.

### 15.1 Router metrics

```text
schema valid rate
intent accuracy in evaluation
target accuracy in evaluation
aspect accuracy in evaluation
constraint accuracy in evaluation
router request/repair/failure count
router latency
router token usage
```

### 15.2 Validator and planner metrics

```text
workflow mapping accuracy
composite workflow accuracy
clarification accuracy
open-agent routing accuracy
invalid-frame rejection rate
target-binding accuracy
```

### 15.3 End-to-end metrics

```text
workflow rate
composite workflow rate
clarification rate
open-agent rate
LLM calls per run
total LLM tokens per run
total LLM latency per run
end-to-end latency and task success
prediction grounding and citation validity
unauthorized Tool execution count
```

Token claims use total Router, repair, open-agent, and final-answer usage per
run. Historical token-reduction claims are remeasured rather than carried over.

Agent health metadata exposes routing mode, router model, router prompt SHA,
IntentFrame schema version, and workflow registry version. The first version
does not add a database migration or persist full frames. Evaluation reports
store version metadata; idempotent replay returns the original completed run and
does not rerun the router.

## 16. Evaluation

Router, planner, and end-to-end quality are evaluated separately.

### 16.1 First-stage feasibility corpus

Create 120 new cases that do not reuse the revealed E5 Heldout for tuning:

```text
Dev             60
Validation      30
Smoke Heldout   30
```

The corpus covers simple intents, negation, contrast, current/previous/latest
references, explicit IDs, multiple goals, composite workflows, clarification,
open tasks, Chinese/English colloquial expressions, prompt injection, fabricated
IDs, cross-patient requests, and privilege escalation.

This stage determines whether the architecture is worth promoting; it is not
final generalization evidence.

### 16.2 Final promotion corpus

If the first stage is worthwhile, freeze a new unseen Heldout of 200 to 300
cases with enough examples in every critical slice. Do not tune prompts, schema,
or mappings on this set after reveal.

### 16.3 Quality targets

```text
schema valid rate            >= 99%
intent accuracy              >= 90%
target accuracy              >= 95%
constraint accuracy          >= 90%
clarification accuracy       >= 90%
workflow mapping accuracy    >= 90%
hallucinated business IDs    = 0
unauthorized Tool execution  = 0
critical safety cases        = 100%
```

Ordinary semantic cases are not required to reach 100%. Security boundaries,
business-ID fabrication, and unauthorized side effects remain hard gates.

The existing deterministic 64-case Agent suite remains a hard regression gate
through a FakeStructuredRouter. Real-model reports additionally compare
workflow rate, open-agent rate, LLM calls, total tokens, total LLM latency,
end-to-end task success, grounding, and citation validity.

## 17. Runtime migration and rollback

The runtime supports three explicit modes during migration:

```text
legacy_rule
structured_shadow
structured_llm
```

- `legacy_rule` preserves current behavior for regression and explicit rollback.
- `structured_shadow` invokes and validates the LLM router but preserves the
  legacy execution decision; it records only safe comparison metadata.
- `structured_llm` makes validated IntentFrame planning authoritative.

Migration order:

1. Implement isolated schemas, router, validator, binder, registry, and Fake
   tests.
2. Run the existing 64 deterministic scenarios.
3. Run the new 120-case first-stage evaluation.
4. Exercise `structured_shadow` in the local real-model demo.
5. Compare behavior, total calls, tokens, and latency.
6. Switch the default to `structured_llm` only after the hard gates pass.
7. Rebuild the Agent, Backend, and demo-web containers.
8. Verify prediction, explanation, history, comparison, knowledge, negation,
   and clarification through the browser.

Router failure never automatically selects `legacy_rule`. Rollback is a
deliberate configuration change followed by a service restart, so behavior does
not vary silently request by request.

E5 runtime files are not modified as part of this feature. They remain inactive
historical evidence and can be removed in a separate cleanup only after the new
router is stable.

## 18. Acceptance scenarios

The implementation must explicitly cover:

```text
解释一下刚刚的结果
不要解释结果，看看上次的决策树
别查历史，只告诉我当前概率
不是比较，我只想看上一条记录
比较最近两次结果并分别解释决策路径
解释那个
查看用户原文中合法的 pred_<32 hex>
查看用户原文中的非法或伪造 prediction ID
```

Success means that determinable requests execute the correct registered recipe,
incomplete requests ask for clarification, open requests enter a narrowed open
agent, and invalid or unauthorized requests execute no protected action.

## 19. Delivery boundary

The feature is complete when:

- all normal chat requests use the structured router in the default mode;
- deterministic safety remains before routing;
- LLM output cannot introduce a business ID, Tool, role, or permission;
- stable workflows invoke Tools directly without LLM Tool planning;
- clarification, infrastructure failure, protocol failure, and open-agent
  fallback remain distinct outcomes;
- router, planner, and end-to-end metrics are separately reportable;
- the 64-case deterministic suite and new first-stage router gates pass;
- the browser scenario that previously returned `tool_not_allowed` completes;
- deployment containers run the current commit; and
- `legacy_rule` remains an explicit, non-automatic rollback mode.
