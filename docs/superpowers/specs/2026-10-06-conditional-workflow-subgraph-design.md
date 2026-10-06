# TreeSem Agent Conditional Workflow / SubGraph Design

> Status: **Living design draft**  
> Branch: `feat/qwen-real-agent-evaluation`  
> Purpose: This document is the design baseline for upgrading the current deterministic linear Workflow into a lightweight state-driven Conditional SubGraph. Future implementation must follow the decisions recorded here. When a design decision changes, update this document first.

## 1. Goal

Keep the current hybrid orchestration architecture:

```text
Structured Intent Router
        ↓
Validation / business-object binding
        ↓
Dispatcher
   ┌───────────────┐
   ↓               ↓
Workflow        Open Agent
```

The change is limited to the **Workflow internal execution model**.

Current model:

```text
WorkflowRecipe
    ↓
Stage 1 → Stage 2 → Stage 3 → END
```

Target model:

```text
WorkflowGraph
    ↓
Node → State update → Conditional Edge → Next Node / END
```

The Workflow remains deterministic in the engineering sense: developers define all allowed Nodes, Edges, and branch conditions in advance. Runtime State determines which predefined path is taken.

The goal is **not** to make Workflow behave like an Open Agent.

---

## 2. Why this change is justified

The existing `Recipe + Stage` model is sufficient for short fixed sequences such as:

```text
history → compare
history → explanation
```

The upgrade is justified only because the planned medical workflow now has genuine runtime branches:

1. Model result differs → later processing may differ.
2. User explicitly requests explanation → explanation may be required even for a negative model result.
3. Evidence is sufficient → generate directly.
4. Clinical background evidence is missing → perform RAG retrieval and re-check.
5. Core model evidence is missing → RAG cannot repair it; degrade or terminate.
6. Tool result is retryable / fatal → retry, fallback, or terminate accordingly.

Therefore the target is a **lightweight Conditional SubGraph**, not a general-purpose graph framework rewrite.

---

## 3. Current implementation baseline

Relevant current components:

- `agent/workflow_registry.py`
  - `WorkflowRecipe`
  - `WorkflowStage`
  - `ArgumentSource`
  - `WorkflowRegistry`
- `agent/deterministic_workflow.py`
  - `DeterministicWorkflowExecutor`
  - currently executes `for stage in recipe.stages`
- `agent/execution_state.py`
  - `ExecutionState`
  - stores trusted Tool execution facts, Evidence, goal completion, and budgets
- `agent/intent_dispatch.py`
  - chooses Workflow / Composite Workflow / Clarification / Open Agent

The top-level Router and Dispatcher architecture should remain intact.

---

## 4. First real Conditional SubGraph

The first SubGraph should be a **risk assessment and explanation workflow** because it has real business branches rather than synthetic graph structure.

Target behavior:

```text
                    Risk Assessment Workflow
                              │
                              ▼
                    [1] Prediction Node
                              │
                              ▼
                    Trusted prediction facts
                 label / positive_probability /
                 confidence / prediction_id
                              │
                              ▼
                    [2] Risk Policy Node
                    deterministic mapping
                              │
                 ┌────────────┴────────────┐
                 │                         │
          MODEL_NEGATIVE              MODEL_POSITIVE
                 │                         │
                 │                         ▼
                 │                 [3] Explanation Node
                 │                         │
                 │                 important_features
                 │                   decision_path
                 │                         │
                 └────────────┐            │
                              ▼            ▼
                       [4] Evidence Check
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
          CORE_EVIDENCE   CLINICAL_EVIDENCE   SUFFICIENT
             MISSING          MISSING            │
                 │              │                 │
                 ▼              ▼                 │
          Fallback / End   [5] Knowledge RAG    │
                                │                 │
                                ▼                 │
                         Evidence Recheck ────────┘
                                │
                         ┌──────┴──────┐
                         │             │
                      SUFFICIENT    STILL MISSING
                         │             │
                         ▼             ▼
                  Final Generate   Degraded Response
                         │             │
                         └──────┬──────┘
                                ▼
                               END
```

### 4.1 Explanation routing rule

The first real conditional rule is:

```text
need_explanation =
    workflow_risk_band == MODEL_POSITIVE
    OR user_requested_explanation
```

Therefore:

- `MODEL_POSITIVE` defaults to explanation.
- `MODEL_NEGATIVE` can skip automatic explanation.
- If the user explicitly requested explanation, `MODEL_NEGATIVE` still enters the Explanation Node.

`user_requested_explanation` must come from the already validated structured intent, not from a second free-form LLM judgment inside the Workflow.

---

## 5. Risk Policy decision

### 5.1 Current model output

The current model does **not** directly produce `LOW / MEDIUM / HIGH` clinical risk levels.

It produces model facts such as:

```text
label
positive_probability
confidence
cluster_id
tree_probability
tree_leaf_id
```

The current binary serving contract uses class `0 / 1` and a binary decision threshold.

### 5.2 V1 policy: binary workflow band

For V1, do not invent unvalidated clinical thresholds.

Use a thin deterministic mapping layer:

```text
label = 0 → MODEL_NEGATIVE
label = 1 → MODEL_POSITIVE
```

The derived field should be named as a Workflow control concept, for example:

```text
workflow_risk_band
```

Allowed values in V1:

```text
UNSET
MODEL_NEGATIVE
MODEL_POSITIVE
```

This is **not** a clinical `low / medium / high` risk classification.

### 5.3 Important semantic boundary

- `positive_probability` is the model output associated with the positive class.
- `confidence` is confidence in the predicted class and must not be interpreted as patient risk probability.
- `tree_probability` should not be used to define the Workflow risk band unless separately validated for that purpose.

---

## 6. Evidence routing

Evidence checking is a first-class Workflow decision.

The Evidence Check must distinguish at least:

```text
UNCHECKED
SUFFICIENT
CORE_EVIDENCE_MISSING
CLINICAL_EVIDENCE_MISSING
```

### 6.1 Core evidence missing

Examples:

- prediction result unavailable
- required prediction id unavailable
- explanation requested but required explanation result unavailable

RAG cannot repair missing model facts.

Route:

```text
CORE_EVIDENCE_MISSING → Fallback / Terminate
```

### 6.2 Clinical evidence missing

The model facts are available, but the user's requested medical-background explanation needs external medical knowledge.

Route:

```text
CLINICAL_EVIDENCE_MISSING
    ↓
Knowledge RAG
    ↓
Evidence Recheck
```

If evidence is still insufficient, produce a degraded response that explicitly limits itself to supported model facts instead of inventing clinical claims.

### 6.3 Evidence sufficient

```text
SUFFICIENT → Final Generate
```

---

## 7. HITL / manual approval decision

V1 will **not** implement `WAITING_APPROVAL` or content approval.

Reason:

- The system is primarily designed as a doctor-facing decision-support system.
- Current capabilities are read-only analysis: prediction, explanation, history comparison, knowledge retrieval, and grounded response generation.
- These operations do not directly produce external side effects.
- The doctor is already the human decision-maker consuming the analysis.

The safety focus for V1 is therefore:

- authorization
- trusted patient / prediction binding
- evidence completeness
- grounding
- constrained Tool execution
- degraded response when evidence is insufficient
- avoiding unsupported clinical claims

HITL should be reconsidered only if future Tools introduce side effects such as:

- writing a medical record
- changing a follow-up plan
- sending a patient notification
- creating a clinical task
- triggering an external alert or action

For those future capabilities, confirmation should be placed **before the side-effecting Tool execution**, not before displaying normal read-only analysis.

---

## 8. State architecture

### 8.1 Decision

Keep the existing `ExecutionState` as the trusted execution-fact ledger and add a separate lightweight `WorkflowState` for graph-control state.

```text
WorkflowState
    │
    ├── execution: ExecutionState
    │      └── trusted Tool facts / Evidence / budgets
    │
    └── graph-control state
           └── current node / derived routing state / termination
```

Do **not** replace the existing `ExecutionState` with one large universal State object.

### 8.2 ExecutionState responsibility

`ExecutionState` answers:

> What actually happened during execution?

It remains responsible for data such as:

```text
ValidatedIntent
Tool usages
Tool attempts
Evidence
Raw Tool facts
History-derived prediction ids
Completed goals
Remaining budgets
```

It is the trusted fact layer.

### 8.3 WorkflowState responsibility

`WorkflowState` answers:

> Given the trusted facts, what control state is the Workflow in and where may it go next?

Initial conceptual fields:

```text
WorkflowState

execution: ExecutionState

workflow_id
workflow_version

current_node
visited_nodes
node_attempts

workflow_risk_band:
    UNSET
    MODEL_NEGATIVE
    MODEL_POSITIVE

evidence_status:
    UNCHECKED
    SUFFICIENT
    CORE_EVIDENCE_MISSING
    CLINICAL_EVIDENCE_MISSING

requested_explanation

last_node_status:
    SUCCESS
    RETRYABLE_ERROR
    FATAL_ERROR

termination_reason
```

Exact Python types and final field list remain to be designed before implementation.

### 8.4 No duplicate prediction state

Do not copy model facts such as the following into `WorkflowState`:

```text
prediction_id
label
positive_probability
confidence
important_features
citations
history
```

Those facts already belong to the execution-fact layer.

If graph logic needs them, expose read-only accessors from `ExecutionState` rather than maintaining duplicate mutable copies.

### 8.5 State update rule

Use the following conceptual separation:

```text
Tool Node
→ primarily produces trusted facts
→ records them in ExecutionState

Policy Node
→ reads trusted facts
→ derives finite Workflow control state
→ writes WorkflowState

Conditional Edge
→ reads derived Workflow control state
→ chooses one predefined next node
```

Example:

```text
Prediction Tool Result
    ↓
ExecutionState
    ↓
Risk Policy Node
    ↓
workflow_risk_band
    ↓
Conditional Edge
```

This keeps business reasoning out of the generic Graph Executor.

---

## 9. Node model

### 9.1 Decision

V1 uses a **single declarative `WorkflowNode` model with a small `NodeKind` enum**, rather than a large inheritance hierarchy or arbitrary Python callbacks.

Executable node kinds in V1:

```text
TOOL
POLICY
```

Graph termination is represented by terminal outcomes such as:

```text
END_SUCCESS
END_DEGRADED
END_FAILURE
```

Terminals are not ordinary executable business nodes.

### 9.2 Tool Node

A Tool Node invokes one registered Tool and produces trusted execution facts.

Examples:

```text
prediction
explanation
knowledge_search
```

Conceptual behavior:

```text
resolve arguments
    ↓
ToolRegistry.execute(...)
    ↓
Tool Result
    ↓
ExecutionState.record(...)
```

A Tool Node must **not** decide which node runs next. It executes and records facts only; graph routing belongs to Edges.

Tool Nodes should preserve the existing explicit argument-binding mechanism where possible, including `tool_name` and `ArgumentSource`.

Conceptual definition:

```text
WorkflowNode
    node_id = "fetch_explanation"
    kind = TOOL
    tool_name = "get_explanation"
    argument_source = BOUND_PREDICTION
```

### 9.3 Policy Node

A Policy Node reads trusted facts and derives finite deterministic Workflow control state. It does not call an external service or an LLM.

Examples:

```text
risk_policy
evidence_check
```

`risk_policy`:

```text
ExecutionState prediction facts
    ↓
label = 0 / 1
    ↓
WorkflowState.workflow_risk_band
```

`evidence_check`:

```text
ValidatedIntent + ExecutionState evidence
    ↓
WorkflowState.evidence_status
```

Do not create a separate `CHECK` node kind for V1. Evidence checks, validation-style derivations, and similar deterministic read-and-derive operations are all `POLICY` nodes.

### 9.4 Why there is no separate Check / Transform / Decision node kind

For V1, the following operations share the same essential contract:

```text
read trusted state
    ↓
perform deterministic program logic
    ↓
write a finite control value
```

Creating separate node kinds such as `CHECK`, `TRANSFORM`, or `DECISION` would add framework abstraction without solving a current business need.

### 9.5 Node and Edge responsibilities must remain separate

A Node defines:

> What work is performed?

An Edge defines:

> Given the resulting control state, where may execution go next?

Therefore a Node must not embed routing logic such as:

```text
execute
if condition:
    next = A
else:
    next = B
```

Instead:

```text
risk_policy Node
    ↓
workflow_risk_band
    ↓
Graph Edge
    ├─ MODEL_POSITIVE → explanation
    └─ MODEL_NEGATIVE → evidence_check
```

This prevents business routing from leaking back into Node handlers or the generic Graph Executor.

### 9.6 Declarative model, not subclass hierarchy

Do not introduce a class hierarchy such as:

```text
BaseNode
 ├─ ToolNode
 ├─ RiskPolicyNode
 ├─ EvidenceCheckNode
 └─ KnowledgeNode
```

Prefer one declarative model such as:

```text
WorkflowNode
    node_id
    kind
    tool_name / argument_source      # TOOL
    policy_handler                   # POLICY
```

Exact field names and validation rules will be finalized with the Graph Registry design.

The reasons are:

- easier static graph validation
- easier graph version hashing
- consistent with the existing declarative `WorkflowRegistry` style
- fewer small framework classes
- clearer separation between node metadata and handler implementation

Arbitrary anonymous callbacks or lambdas should not be used as Node definitions because they weaken validation, introspection, versioning, and testability.

### 9.7 Final response generation is outside the graph in V1

Do not add `FinalGenerateNode`, generic LLM nodes, or response-rendering nodes to the V1 SubGraph.

The graph is responsible for preparing the trusted facts and Evidence required to answer the request and then ending with a terminal outcome.

```text
Conditional Workflow
       ↓
trusted facts + Evidence ready
       ↓
END_SUCCESS / END_DEGRADED / END_FAILURE
       ↓
existing Final Response layer
       ↓
response to doctor
```

This limits the scope of the graph migration and preserves the existing response-policy / grounding layer.

### 9.8 Initial V1 business nodes

The first Conditional SubGraph is expected to use:

```text
TOOL
────
prediction
explanation
knowledge_search

POLICY
──────
risk_policy
evidence_check

TERMINAL
────────
END_SUCCESS
END_DEGRADED
END_FAILURE
```

The same `evidence_check` Policy Node may be revisited after knowledge retrieval. The Graph Executor must therefore later define bounded node-attempt / cycle protection so the graph cannot loop indefinitely.

---

## 10. Tool error routing

Tool outcomes should eventually become explicit graph-routing states rather than being handled only as an early return from the linear executor.

Minimum control categories:

```text
SUCCESS
RETRYABLE_ERROR
FATAL_ERROR
```

Intended routing:

```text
SUCCESS         → normal next node
RETRYABLE_ERROR → bounded retry where policy permits
FATAL_ERROR     → fallback / terminate
```

Retry rules must preserve current engineering constraints:

- read-only transient failures may be retried within a small bound
- invalid arguments, permission errors, and failed business preconditions should not be blindly retried
- side-effecting operations, if introduced later, require stricter idempotency / unknown-outcome handling

The exact error classifier and retry-edge representation are still open design questions.

---

## 11. V1 scope

V1 should implement enough graph behavior to solve real business branching without becoming a general graph framework.

Required V1 capabilities:

1. Explicit Node identity.
2. Explicit normal and conditional Edges.
3. Runtime next-node resolution from `WorkflowState`.
4. `WorkflowState` layered over existing `ExecutionState`.
5. `TOOL` and `POLICY` executable node kinds plus terminal outcomes.
6. Risk Policy branch (`MODEL_NEGATIVE / MODEL_POSITIVE`).
7. User-requested explanation branch.
8. Evidence branch (`SUFFICIENT / CORE_EVIDENCE_MISSING / CLINICAL_EVIDENCE_MISSING`).
9. RAG evidence-repair branch.
10. Explicit success / retryable / fatal Tool outcome path, at least for the first SubGraph.
11. Bounded execution protection such as node/tool-call limits and termination reason.

---

## 12. Explicit non-goals for V1

Do not add these merely to make the system look more like LangGraph:

- general-purpose graph DSL
- arbitrary dynamic node creation
- LLM-created edges
- multi-agent graph orchestration
- persistent checkpoint / pause / resume
- content approval / manual review
- side-effect action confirmation
- multi-reviewer approval chains
- low / medium / high clinical-risk thresholds without validation
- replacing the top-level Router / Dispatcher architecture
- migrating the entire project to LangGraph
- a large Node subclass hierarchy
- generic LLM / final-response nodes inside the Workflow graph

---

## 13. Design principles agreed so far

1. **The graph is predefined; the runtime path is state-dependent.**
2. **Deterministic facts stay in program-controlled state.**
3. **LLM freedom is introduced only where semantic uncertainty requires it.**
4. **ExecutionState is the fact layer; WorkflowState is the control layer.**
5. **Tool Nodes create trusted facts; Policy Nodes derive finite control state; Edges perform routing.**
6. **Nodes never choose their own next node.**
7. **RAG repairs missing knowledge evidence, not missing model facts.**
8. **No artificial HITL for read-only doctor-facing analysis.**
9. **Do not redesign the whole Agent around a graph framework.**
10. **The upgrade must solve actual branching needs, not merely rename linear stages as nodes.**
11. **Final natural-language generation remains outside the Workflow graph in V1.**

---

## 14. Open design questions

The following must be resolved and recorded here before implementation:

1. **Edge model**
   - Exact normal-edge and conditional-edge representation.
   - Whether conditions are registered Python functions, enums/policies, or another constrained representation.

2. **Argument binding**
   - How much of the existing `ArgumentSource` mechanism remains.
   - How Nodes access results from `ExecutionState` without duplicating data.

3. **Graph Executor**
   - Exact execution loop.
   - Node-attempt limits, cycle handling, deadline and tool-call budgets.

4. **Backward compatibility**
   - Whether current linear `WorkflowRecipe` definitions are migrated all at once or adapted into the new graph representation.

5. **Error routing**
   - Exact retryable/fatal classification and bounded retry behavior.

6. **Testing**
   - Graph validation tests.
   - Branch-path tests.
   - Existing deterministic Workflow regression tests.
   - Evidence-routing and failure-path tests.

---

## 15. Implementation gate

No product-code implementation should begin until:

1. the remaining open design questions are resolved;
2. this document is updated to reflect those decisions;
3. the final design is reviewed and approved;
4. a concrete implementation plan is written from this document.

When implementation begins, this document is the source of truth for intended behavior and architectural boundaries.
