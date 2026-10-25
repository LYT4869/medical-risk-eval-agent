# TreeSem V2 M3｜单 Investigation Agent + 动态 Agenda（P001 SIM）

## Status and critical boundary

This is **real LangGraph implementation** of an issue-driven Investigation Agent, with a fully deterministic **FakeInvestigationAgent** regression path and a configurable **OpenAI-compatible model decision adapter**. The GitHub CI tests **do not run Qwen or any deployed LLM**. Thus successful tests establish *runtime correctness under a controlled Agent decision trace*, not real model reliability or generalized medical reasoning.

Dependency judgement: M3 begins **after** an explicitly confirmed Task Contract and checks that the authoritative C++ MySQL Run remains confirmed for its revision. Missing M1 Gateway support for *revision/clarification/reconfirm* does **not** block this investigation runtime; those edits are a separate upstream workflow and will be implemented **immediately after M3** (user-requested order). The M3 module does not attempt to bypass confirmation or create a second business Run.

## What actually runs

```text
M1 confirmed Contract (MySQL Run status=contract_confirmed)
     -> M3 seed Agenda [I-01 prior history, I-02 sources, I-03 model]
     -> one Investigation Agent proposes next Action (Fake or local LLM)
           ↓ Guard: schema, allowed Tool, Issue-specific record IDs, budget,
                    no duplicate reads, source snapshot time, patient/Run
     -> Tool (get_patient_timeline or read_patient_record)
           ↓
        MySQL SIM Evidence (source id + immutable logical_id/version)
     -> Agent proposes Close (Issue evidence_ids, followup tag)
           ↓ program checks referenced saved Evidence and required disclosures
        MySQL immutable IssueResult + Operation Journal
           ↓
        I-02 encounters E-002(original 250mL)/E-003(referral ~800mL)
        -> proposal followup='source_discrepancy'
           ↓
        Validator adds I-04, stable ID and de-dup, without replacing I-02
           ↓
        I-03 blocked (unverified prenatal model): no model call or fake score
        I-04 closed_unresolved (both source estimates remain)
           ↓
        restricted, unsigned, template-only six-section Draft v1
        -> MySQL immutable draft / doctor sees the named report
        -> LangGraph END (no mandatory review interrupt)
```

The model does **not** directly write diagnosis text, SQL, doctor identity, or arbitrary Tool arguments. All synthetic IssueResult sentences are deterministic source-bounded templates. Agent output is limited to `action=tool/close`, `tool`, `record_id`, `evidence_ids`, and optional controlled `followup` enum. The runtime uses one Agent (Fake or OpenAI-compatible) both for Tool proposals and for Issue close/add-followup proposals; there is no separate Planner/Replanner/Memory Agent.

**Agenda State** in official SqliteSaver contains Issue IDs, fixed topic enums, todo/done/blocked statuses, Evidence ID refs, immutable result keys and quotas. It does not contain raw clinical source records or capability tokens. On each model call the Working Context is rebuilt from the current Issue, the confirmed Contract goal, Skill/Risk Profile identifiers and at most three source-bounded synthetic Evidence projections. M3's initial three known P001 Issue categories are seeded deterministically from a fixed Skill policy; the LLM is **not** tested as a free-form initial Agenda planner. I-04 is a runtime-proposed, program-validated dynamic addition.

Tool catalog is deliberately small (two fixed tools, direct selection); adding a Tool Discovery LLM or namespace navigation for only two Tools would be unnecessary. The Tool Adapter permits only P001 / E-001–E-003 at *historical T1 = 2026-10-12*. It excludes E-004/E-005 LIS correction: **M4** will handle revised source event EV-001 and stale propagation. The final Draft is labeled `source_as_of=2026-10-12` and `source_validity=historical_T1_not_current`. It must never be displayed as an up-to-date chart.

## Durability/Guard contract

The Graph deliberately splits **decide** and **execute** into separately checkpointed nodes. A model might produce a different decision on replay, so the accepted, strictly ID-only proposal is checkpointed before any Tool/IssueResult business effect. The executor uses a deterministic operation ID derived from Run + kind + artifact_key and an immutable request fingerprint. If SQL commit succeeds but Graph checkpoint does not, retry reuses the same planned operation rather than calling the LLM again and inventing a new action.

- Maximum **16 decisions** (initial default) and **5 Tool calls**; each action passes pre-Tool Run/Contract/assignment reauthorization (C++ Gateway) plus P001 Tool allowlist/record time boundary. Invalid proposals increment `no_progress` and terminate safely after two consecutive violations; no Report is persisted for incomplete Agenda.
- `I-01` requires E-001; `I-02` requires E-002 and E-003 plus explicit discrepancy followup; `I-03` closes `blocked` without a model call; `I-04` links E-002 and E-003. No diagnosis, no hallucinated measured "actual 800mL", no unsupported PPH label.
- Every closed Issue has an immutable `I-XX:r1` MySQL IssueResult with statement-level source IDs. Updating a closed snapshot is forbidden. `I-04` addition is computed as a deterministic Graph state delta only after idempotent I-02 IssueResult SQL commit. It is **not a single distributed transaction** across MySQL and Checkpointer; the persistent operation journal ensures replay safety.
- C++ Gateway still handles authentic doctor/patient Run authorization and short-lived scoped JWT; M3 never authenticates patients using fake scenario IDs. Local Fake path deliberately uses synthetic `DOC-001/P001` allowlist only.
- The report remains a static M2-derived six-section template, with only source-ID validation. **M5** must replace it with structured LLM Writer, full Coverage and sentence/assertion-level Grounding before any claim about medically supported report correctness.

## Run locally with Fake Agent

```bash
python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
export PYTHONPATH=PythonServices/TreeSemAgent

# M1 local demo remains the upstream explicit confirmation gate
python -m v2_m1.cli --data-dir /tmp/treesem-m3 start
python -m v2_m1.cli --data-dir /tmp/treesem-m3 confirm RUN_ID \
  --decision confirm --revision 0 --contract-revision 1

python -m v2_m3.cli --data-dir /tmp/treesem-m3 RUN_ID
python -m v2_m3.cli --data-dir /tmp/treesem-m3 RUN_ID
```

For C++/MySQL, start the native gateway in **SIM-only** mode with MySQL migrations 001–007 and a real doctor-confirmed *synthetic* Run. Put the ephemeral `capability_token` from the authorized doctor API in the environment variable `TREESEM_M3_CAPABILITY_TOKEN` and set `TREESEM_M3_GATEWAY_URL=https://your-secured-gateway`. Then:

```bash
python -m v2_m3.cli --data-dir /tmp/treesem-m3-gateway \
  --mode gateway RUN_ID --contract-revision 1
```

For **opt-in local inference**, use `--agent local_llm` and configure `TREESEM_M3_LLM_ENDPOINT` (a reachable locally hosted OpenAI-compatible `/v1/chat/completions` route), `TREESEM_M3_LLM_MODEL`, optionally `TREESEM_M3_LLM_TOKEN`. The Agent outputs a JSON action proposal, and the same Python Guard validates it before side effects. This provides an integration option but the repo CI does **not** provide or measure a locally deployed Qwen server, so there are **no real LLM success/latency/token statistics**. Do not feed patient PHI to an unapproved endpoint.

## Acceptance and test scope

```bash
python -m unittest discover \
  -s PythonServices/TreeSemAgent/v2_m3/spec_tests -p 'test_*.py' -v
```

Additional GitHub Actions `v2_m3_mysql_integration` builds a real C++ Gateway with `KAMA_ENABLE_MYSQL=ON` and JWT, starts **ephemeral MySQL 8**, creates synthetic doctor/patient authorization, resumes M1 HITL, runs M3's dynamic Agenda, checks I-04 and 8 persisted Artifacts (3 Evidence + 4 IssueResult + 1 Draft), tests separate Run scoped capability and crash-window replay by committing E-001 before Graph first executes. Check the exact workflow run's result before asserting passing tests.

**After M3, next prescribed work is M1 Gateway Contract revision/clarification/reconfirm** — before M4. Remaining M3 research validation: exercise a real locally deployed compatible LLM over a larger multi-case set and record Tool choice, Issue quality, errors, budget and latency, rather than turning Fake P001 behavior into a success percentage. Project milestones remain stacked Draft PRs; none are merged to `main`.
