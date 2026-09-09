# treeSem Agent Rule Taxonomy and Structured Evidence Design

## Status

Approved direction from the 2026-09-09 routing review. This is a bounded
refinement of the existing rule-first Agent routing path. It does not promote
the E5 semantic candidate, change deterministic Workflow stages, or add an
extra LLM routing call.

## Objective

Turn the current partially structured `RuleRouter` into an explicit two-step
business-intent component:

```text
normalized message
  -> typed action/object/reference evidence
  -> taxonomy resolver
  -> high-precision business scope or UNKNOWN
```

The change makes the existing routing policy easier to explain, test, and
measure. It does not attempt to classify every natural-language expression.

## Non-goals

- Do not modify the treeSem clinical model or its Serving Bundle.
- Do not replace `multilingual-e5-small` or retune its frozen thresholds.
- Do not make `hybrid_optional` the default.
- Do not train a supervised classifier or add an LLM Router.
- Do not build a rule DSL, routing microservice, planner DAG, or new executor.
- Do not change Tool implementations, RBAC, Capability JWTs, or Skill loading.
- Do not edit the frozen 120/240 Routing Quality Set to improve reported
  metrics.

## Existing baseline

The current implementation is more than one-keyword dispatch. It already
combines task terms with stored-record context, explanation details, knowledge
domain markers, and Workflow dependencies. However, signal extraction and
scope resolution are mixed in one method, and the intermediate evidence is
not inspectable.

The default path remains:

```text
SafetyPolicy
  -> RuleRouter
  -> deterministic Workflow on a rule hit
  -> guarded Open Agent on a miss, ambiguity, or composition
```

The optional Hybrid path remains:

```text
SafetyPolicy
  -> RuleRouter
  -> bounded FP32 ONNX semantic fallback on a rule miss
  -> guarded Open Agent on uncertainty or failure
```

## Taxonomy contract

The six business scopes retain their existing meaning:

| Scope | Primary user goal | Required evidence | Workflow dependencies that do not create a second intent |
|---|---|---|---|
| `prediction` | Run a new demo prediction | prediction action plus demo/sample object or explicit sample reference | none |
| `summary` | Read stored prediction facts | read action, prediction-fact object, and stored prediction reference | `get_prediction` |
| `explanation` | Explain an existing prediction | explanation action/detail plus stored prediction reference | an explicitly requested summary of the same prediction |
| `history` | List saved predictions | list/read action plus history object | none |
| `comparison` | Compare two saved predictions | comparison action plus multiple/prior prediction reference | loading history and reading facts for the compared records |
| `knowledge` | Retrieve general model or PPH knowledge | knowledge/retrieval action plus model/clinical domain object | MCP retrieval |

Dependency absorption is deliberately narrow:

- `comparison` may absorb `history` and `summary` only when they identify or
  display the same records being compared.
- `explanation` may absorb `summary` only when both refer to the same stored
  prediction.
- `prediction + explanation`, `comparison + explanation`,
  `comparison + knowledge`, `history + prediction`, and other independent
  deliverables remain compositional and return `UNKNOWN`.
- Missing references, unsupported goals, or evidence insufficient for one
  scope return `UNKNOWN` rather than guessing.

## Typed evidence model

Add dependency-neutral evidence types:

```python
class IntentAction(str, Enum):
    PREDICT = "predict"
    READ = "read"
    EXPLAIN = "explain"
    LIST = "list"
    COMPARE = "compare"
    RETRIEVE = "retrieve"

class IntentObject(str, Enum):
    DEMO_SAMPLE = "demo_sample"
    PREDICTION_FACT = "prediction_fact"
    EXPLANATION_DETAIL = "explanation_detail"
    HISTORY = "history"
    KNOWLEDGE = "knowledge"

class IntentReference(str, Enum):
    EXPLICIT_SAMPLE = "explicit_sample"
    CURRENT_PREDICTION = "current_prediction"
    PRIOR_PREDICTION = "prior_prediction"
    MULTIPLE_PREDICTIONS = "multiple_predictions"

@dataclass(frozen=True)
class RuleEvidence:
    actions: frozenset[IntentAction]
    objects: frozenset[IntentObject]
    references: frozenset[IntentReference]
    registry_scopes: frozenset[RequestScope]
    explicit_skill: bool = False
    vague_reference: bool = False
```

`RuleEvidenceExtractor` owns normalization and finite lexical marker groups.
It may use `TaskRegistry.rule_terms` as coarse scope evidence, but a registry
term alone is not sufficient for context-sensitive scopes.

`RuleIntentResolver` owns the taxonomy table, dependency absorption, and
compositional fallback. It does not inspect raw text.

`RuleRouter` becomes a small facade:

```python
evidence = extractor.extract(message)
return resolver.resolve(evidence)
```

No general boolean expression language is added to YAML. Complex predicates
remain typed Python code and unit tested.

## Safety boundary

`SafetyPolicy` already runs before business routing and remains a separate
component. This change does not expand its phrase lists. Security remains
layered:

```text
natural-language risk prefilter
  -> business routing
  -> Tool allowlist and Schema
  -> C++ RBAC / Capability / resource ownership
```

The historical Heldout `safety_accuracy=20%` measures the language prefilter,
not execution authorization. Improving the risk classifier is a separate
future cycle with its own risk taxonomy and independent data.

## Evaluation contract

The evaluator must separate these outcomes:

- correct deterministic route;
- abstention on a known intent;
- wrong deterministic Workflow for a known intent;
- forced deterministic route for Unknown/OOD;
- forced deterministic route for a compositional request;
- incorrect safety prefilter result.

Add additive metrics while retaining current JSON fields:

```text
deterministic_precision
deterministic_coverage
known_abstention_rate
known_misroute_rate
unknown_forced_route_rate
compositional_forced_route_rate
per_scope_precision_recall
```

The existing `rule_precision` and `rule_coverage` fields remain for artifact
compatibility but are documented as legacy names. Hybrid reports use the new
`deterministic_*` names because their routed results can come from either Rule
or Semantic sources.

## Test data policy

- Existing unit tests remain normal regression inputs.
- Add a small taxonomy contract suite with canonical dependency and true
  multi-intent examples. It proves implementation semantics, not statistical
  generalization.
- Do not modify the frozen 360-case quality set.
- The already-viewed 240 Heldout split may be reported only as historical
  diagnostic evidence and must not be used to claim a new promotion.
- A later Router promotion requires a new independent
  train/Calibration/Heldout cycle based on unseen or real traffic expressions.

## Compatibility

- `AgentRouter.route(message) -> RoutingDecision` is unchanged.
- `RuleRouter.route(message) -> RoutingDecision | None` is unchanged.
- `rule`, `hybrid_optional`, and `hybrid_required` configuration values are
  unchanged.
- Default routing mode remains `rule`.
- Workflow stages and Tool permissions remain unchanged.
- Existing semantic Artifact stays usable because Task Registry and semantic
  thresholds are not modified in this cycle.

## Completion criteria

- Raw marker extraction and scope resolution are independently testable.
- Canonical dependency cases route to one deterministic Workflow.
- Independent multi-intent cases return `UNKNOWN`.
- Rule misses still enter Semantic fallback in Hybrid mode and Open Agent in
  Rule mode.
- New evaluation metrics distinguish abstention from harmful misrouting.
- Existing frozen evaluation artifacts are not rewritten.
- Full Agent, Knowledge, C++ and Adapter regressions pass.
- Documentation states that the Router is structured lexical evidence, not a
  learned semantic parser.
