# Agent Tool Governance Design

## Context

The treeSem Agent currently asks an OpenAI-compatible model to choose from the
full native Tool, knowledge Tool, and Skill activation catalog. Prompt rules ask
the model to use the minimum sufficient Tool set, but the runtime enforces only
global step and Tool-call limits plus exact consecutive-call detection.

The single-pass 64-case evaluation exposed a model portability problem:
`qwen-plus` completed 55/64 cases while `qwen3.7-plus-2026-05-26` completed
36/64. The newer model made 222 LLM requests and consumed 489,683 Tokens,
including repeated knowledge searches, unnecessary Skill activation, extra
read-only Tool calls, and failures that the evaluator reported without a useful
internal reason. The evaluation runner also duplicates the previous assistant
message in multi-turn context.

The system must not depend on a model voluntarily following advisory prompt
language for authorization, call budgets, or termination.

## Goals

- Preserve a genuine Agent Loop: the LLM still chooses among the Tools allowed
  for the current request, supplies Tool arguments, and writes the final answer.
- Move Tool visibility, repeat limits, Skill eligibility, and obvious security
  refusal into deterministic code.
- Prevent a successful knowledge lookup from being repeated for the same run.
- Keep one controlled retry when knowledge retrieval fails or returns no evidence.
- Preserve the current public Python and C++ HTTP contracts.
- Produce safe, low-cardinality internal failure codes without exposing prompts,
  Tool arguments, upstream bodies, or patient content.
- Repair multi-turn evaluation context and make real-model failures diagnosable.

## Non-goals

- Streaming responses, asynchronous Run polling, semantic caches, or a new UI.
- Replacing the Agent with a fully deterministic workflow engine.
- Parallel Tool execution.
- Changing treeSem predictions, ONNX serving, MySQL, RBAC, MCP transport, or the
  trusted Skill package format.
- Running another 64-case real-model evaluation in this batch.

## Considered approaches

### Prompt-only tuning

This is the smallest change, but it leaves correctness dependent on model
interpretation. The qwen3.7 result already demonstrates that prompt compliance
does not transfer reliably between model families. This approach is rejected.

### Full deterministic state machine

Every intent and Tool transition would be encoded in application code. This is
predictable but removes too much Agent autonomy, requires broad natural-language
intent coverage, and is unnecessary for the current three Skills. This remains
a fallback if hybrid governance cannot meet the evaluation gates.

### Hybrid deterministic governance

The recommended approach adds a small run-scoped policy around the existing
Agent Loop. Code determines the safe Tool scope and call budget; the model makes
decisions inside that scope. This gives model portability without turning the
project into a fixed command router.

## Architecture

Add an `AgentRunGuard` owned by one `AgentLoop.run()` invocation. It has no
shared mutable state and records only low-sensitivity control state:

```text
request message + actor role
        ↓
AgentRunGuard initial scope
        ↓
LLM sees filtered Tool definitions
        ↓
guard validates proposed Tool calls
        ↓
Tool Registry executes authorized calls
        ↓
guard records outcome and narrows the next scope
```

The guard does not execute Tools and does not inspect Tool response prose. It
uses Tool name, success/error status, result count, active Skill, and whether
verified citations were returned.

### Coarse intent scope

`AgentRunGuard` classifies only high-confidence bilingual request shapes:

- explicit Skill/workflow request;
- demo prediction;
- stored prediction explanation;
- history or comparison;
- model/medical knowledge;
- explicit fabrication, authorization bypass, or other-patient access;
- unknown/general instruction.

Recognized scopes expose a minimal safe Tool set:

| Scope | Initially visible Tools |
|---|---|
| Explicit Skill | `activate_skill` only |
| Demo prediction | `predict_sample` |
| Explanation | `get_explanation`, plus `get_prediction` for summary facts |
| History | `get_prediction_history` |
| Comparison | `get_prediction_history`, `compare_predictions` |
| Knowledge | `search_medical_knowledge` |
| Explicit security abuse | none; return deterministic refusal |
| Unknown | read-only native Tools; no `predict_sample` or `activate_skill` |

Classification is a safety optimization, not the only authorization boundary.
Existing capability tokens, RBAC, Tool schemas, and response grounding remain
authoritative. If a request is not confidently classified, the guard uses the
conservative unknown scope rather than guessing a side-effecting intent.

Skill activation is visible only when the user explicitly asks for a Skill,
workflow, stable process, or equivalent Chinese wording. After activation, the
existing Skill-required Tool set remains the maximum scope.

### Stateful Tool transitions

- After a successful knowledge search returns at least one verified citation,
  `search_medical_knowledge` is removed for the rest of the run.
- If knowledge search fails or returns zero results, one additional search is
  allowed. A third attempt returns a structured Tool error without calling MCP.
- Exact repeated calls remain forbidden. The guard additionally prevents repeat
  knowledge searches after evidence satisfaction even when the query differs.
- After `compare_predictions` succeeds, comparison/history Tools are removed;
  the model must answer rather than reread the same records.
- An activation call cannot be batched with domain Tools. Instead of crashing the
  Run, the activation is processed alone and every batched domain call receives
  a structured `skill_activation_required_first` Tool error, allowing one model
  correction within the existing deadline.
- Tool visibility never expands beyond the request scope, active Skill, actor
  capability, and registered Tool set intersection.

### Deterministic security refusal

High-confidence requests that explicitly ask to fabricate predictions or
citations, bypass authorization, ignore system rules, or access another
patient's records return the existing safe refusal before the first LLM call.
The precheck covers explicit abuse only. Ambiguous requests still enter the LLM
and remain subject to capability validation and response grounding.

No clinical diagnosis or emergency triage is moved into keyword rules in this
batch. Emergency behavior remains governed by the existing response policy and
is included in targeted regression tests.

## Failure model and observability

`AgentExecutionError` gains a stable internal `code`. Required codes are:

```text
llm_failed
agent_timeout
repeated_tool_call
skill_activation_conflict
skill_activation_limit
tool_call_limit
step_limit
tool_not_allowed
knowledge_attempt_limit
```

Public HTTP behavior remains `agent_execution_failed` or `agent_timeout` as it
is today. Internal Trace and evaluation reports may include only the stable code.
They must not include exception text, prompts, model response bodies, Tool
arguments, chat content, tokens, or patient data.

The evaluator will:

- append one user and one assistant message per completed turn;
- retain the first safe internal failure code at scenario level;
- distinguish execution failure from workflow mismatch;
- keep raw qwen-plus and qwen3.7 reports immutable.

## Testing

All behavior changes follow red-green-refactor.

Unit tests must prove:

- one assistant message is retained per completed evaluation turn;
- explicit abuse returns a refusal with zero LLM and Tool calls;
- ordinary knowledge requests cannot activate a Skill;
- explicit workflow requests initially expose only `activate_skill`;
- successful knowledge evidence removes knowledge search from the next turn;
- empty/error knowledge results permit exactly one retry;
- a third knowledge attempt is rejected without MCP access;
- successful comparison prevents redundant history/comparison reads;
- mixed Skill activation and domain calls recover through structured Tool errors;
- every Agent termination path emits the expected safe internal code;
- no new error field reaches the public response model.

Regression gates:

- all TreeSemAgent unit tests pass;
- deterministic evaluation remains 64/64 across 79 turns;
- existing Grounding, Citation, Skill, RBAC, and medical-boundary tests pass;
- `git diff --check` passes.

## Real-model validation

After deterministic gates pass, run one qwen3.7 targeted evaluation with at
most 12 cases and an expected budget of 80,000 to 120,000 Tokens. The set covers
history, comparison, model RAG, clinical RAG, comparison Skill, education Skill,
emergency response, invalid Tool response, explicit abuse, and the previously
stable explanation Skill.

Promotion gates are:

- all security, medical-boundary, Prediction Grounding, and Citation checks pass;
- no successful knowledge query is followed by another knowledge query;
- no ordinary request activates a Skill;
- no unexplained execution failure remains;
- at least 11/12 tasks succeed.

Failure to meet these gates keeps `qwen-plus-2025-07-28` as the default. It does
not trigger another full 64-case run.

## Rollout and compatibility

The guard is enabled for both real and scripted LLM clients so CI exercises the
same orchestration. No environment flag is added in the first batch: safety and
call-budget behavior must not silently differ by deployment. Existing global
step, Tool-call, timeout, scheduler, and capability limits remain in force.

The change is internal to the Python Agent. C++ Gateway requests, Agent
responses, persisted Run summaries, model serving, and MCP schemas remain
backward compatible.
