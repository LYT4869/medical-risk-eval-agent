# Agent Routing and Safety Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Task Registry single-source integration, generalize deterministic safety and compositional routing on the frozen calibration split, and perform one untouched held-out promotion evaluation.

**Architecture:** User input first passes a deterministic concept-based safety policy, then a high-precision rule fast path, then the bounded FP32 ONNX semantic fallback. Static business routing metadata comes from Task Registry; runtime authorization remains the intersection of Registry tools, RBAC/capability scope, Skill narrowing, and per-run state. Ambiguous, multi-intent, failed, timed-out, or overloaded routing falls back to the guarded Open Agent.

**Tech Stack:** Python 3.10, unittest, YAML Task Registry, ONNX Runtime 1.20.1, multilingual E5 routing artifact.

**Spec:** `/home/data/liyingting/models/TRI_VAE/AIModelling/updateplan.md` plus the approved 2026-09-09 routing and safety closure design in chat.

## Global Constraints

- Do not inspect or execute the 240-case held-out split before calibration code and thresholds are frozen.
- Semantic routing may select only registered business scopes; it cannot grant permissions or emit safety decisions.
- Security, medical refusal, RBAC, capability checks, retry budgets, and Skill manifests remain outside Task Registry.
- Do not add a general LLM planner, workflow DSL, second general-purpose LLM, or dynamic plugin system.
- Historical `trivae` source and artifact names remain unchanged; new names use treeSem.
- Use test-first red-green-refactor for every behavior change.

---

### Task 1: Make Task Registry the static business-policy source

**Files:**
- Modify: `PythonServices/TreeSemAgent/agent/run_guard.py`
- Modify: `PythonServices/TreeSemAgent/agent/loop.py`
- Test: `PythonServices/TreeSemAgent/tests/test_run_guard.py`
- Test: `PythonServices/TreeSemAgent/tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `TaskRegistry.definition(scope).allowed_tools`
- Produces: `AgentRunGuard.for_scope(scope, registry=..., include_summary=...)`

- [ ] Add a failing test proving a custom Registry changes initial business tools without editing the Guard.
- [ ] Run the focused test and verify it fails because the Guard still uses `_INITIAL_TOOLS`.
- [ ] Inject Task Registry into Guard construction while preserving special scopes and runtime narrowing.
- [ ] Run Guard, Workflow, Task Registry, and Agent Loop tests.
- [ ] Commit the independently testable change.

### Task 2: Replace phrase patches with concept-based deterministic safety policy

**Files:**
- Create: `PythonServices/TreeSemAgent/agent/safety_policy.py`
- Modify: `PythonServices/TreeSemAgent/agent/routing.py`
- Modify: `PythonServices/TreeSemAgent/agent/policy.py`
- Test: `PythonServices/TreeSemAgent/tests/test_routing.py`
- Test: `PythonServices/TreeSemAgent/tests/test_run_guard.py`
- Test: `PythonServices/TreeSemAgent/tests/test_agent_loop.py`

**Interfaces:**
- Produces: `SafetyPolicy.evaluate(message) -> SafetyDecision`
- Preserves: safety reason codes and fixed no-LLM/no-Tool refusal behavior

- [ ] Add failing parameterized tests from calibration-only security-abuse categories and matched benign contrast cases.
- [ ] Add failing tests for personalized medication changes, procedure decisions, diagnostic certainty, treatment guarantees, fabricated evidence, and current emergencies.
- [ ] Verify each new test fails for the missing concept rather than malformed fixtures.
- [ ] Implement normalized concept groups and conjunction rules; do not reject on a single medical noun.
- [ ] Preserve defensive, educational, hypothetical, and system-security explanation requests.
- [ ] Run safety, Guard, Agent Loop, and evaluation tests.
- [ ] Commit the independently testable change.

### Task 3: Generalize compositional request fallback

**Files:**
- Modify: `PythonServices/TreeSemAgent/agent/routing.py`
- Test: `PythonServices/TreeSemAgent/tests/test_routing.py`
- Test: `PythonServices/TreeSemAgent/tests/test_workflow.py`
- Test: `PythonServices/TreeSemAgent/tests/test_agent_loop.py`

**Interfaces:**
- Produces: deterministic `UNKNOWN/rule_compositional` for independent multi-task requests
- Preserves: summary-with-explanation and history-before-comparison workflow dependencies

- [ ] Add failing tests from calibration-only two-intent and three-intent formulations.
- [ ] Add contrast tests for phrases that are steps of one supported workflow.
- [ ] Implement intent evidence collection and dependency-aware compositional classification.
- [ ] Run Rule, Workflow, and Agent Loop tests.
- [ ] Commit the independently testable change.

### Task 4: Recalibrate the bounded FP32 ONNX semantic fallback

**Files:**
- Modify: `PythonServices/TreeSemAgent/config/tasks.yaml` only when calibration evidence supports new intent prototypes
- Modify: `PythonServices/TreeSemAgent/config/routing_thresholds.json`
- Modify: routing artifact under the ignored local artifact directory
- Test: `PythonServices/TreeSemAgent/tests/test_routing_calibration.py`
- Test: `PythonServices/TreeSemAgent/tests/test_routing_backend_parity.py`

**Interfaces:**
- Consumes: frozen 120-case calibration split only
- Produces: calibrated similarity, margin, and secondary-intent thresholds

- [ ] Run rule and hybrid calibration reports without loading held-out messages.
- [ ] Add tests for any newly discovered calibration boundary.
- [ ] Update intent prototypes by concept, without copying evaluation messages.
- [ ] Re-export backend-aligned FP32 intent vectors and calibrate thresholds.
- [ ] Verify safety recall 100%, benign safety false positives 0, unknown recall at least 95%, compositional fallback at least 95%, and meaningful known-intent improvement.
- [ ] If deterministic safety still misses the gate, stop and design a separate one-way safety risk classifier; do not silently add an LLM guard.
- [ ] Commit configuration and reproducibility changes.

### Task 5: Freeze and run the one-shot held-out promotion gate

**Files:**
- Create: `docs/reports/agent-routing-safety-closure.md`
- Modify: `docs/reports/agent-routing-evaluation.md`
- Modify: `docs/treesem-interview-guide.md`

**Interfaces:**
- Consumes: frozen code, Registry, thresholds, and FP32 ONNX artifact checksums
- Produces: immutable Rule-versus-Hybrid held-out report and promotion decision

- [ ] Record Git SHA and all policy/config/artifact checksums before evaluation.
- [ ] Run the 240 held-out cases exactly once.
- [ ] Do not tune the evaluated version from held-out failures.
- [ ] Promote Hybrid only if safety is 100%, benign false refusal is 0, unknown/compositional recall are at least 95%, business Macro F1 improves, and known routing improves materially.
- [ ] Keep Rule as default if any hard gate fails.
- [ ] Document one safety and one compositional bad-case story without sensitive content.

### Task 6: Full regression and final evidence

**Files:**
- Modify only documentation or tests required by verified regressions.

**Interfaces:**
- Produces: final deterministic Agent, RAG, Skill, RBAC, concurrency, and targeted real-LLM evidence

- [ ] Run all routing unit tests.
- [ ] Run the 64-case deterministic Agent evaluation.
- [ ] Run RAG, Skill, RBAC/capability, and no-sensitive-log tests.
- [ ] Run routing executor concurrency tests for 20 rounds.
- [ ] Run only targeted paid-LLM smoke cases after deterministic gates pass.
- [ ] Run `git diff --check` and inspect the final diff for held-out leakage or duplicated policy configuration.
- [ ] Commit the final reports and documentation.
