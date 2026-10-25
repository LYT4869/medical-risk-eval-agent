# TreeSem V2 M2-A — Synthetic Level-0 vertical slice

**Status: implementation prototype.** The P001 Python M2 path uses a **separate local SQLite fake ArtifactRepository**, and the C++/MySQL SIM Artifact API is an **independent authenticated persistence boundary**. These two codepaths are **not yet connected end to end**; do not claim this is the finished C++→MySQL→Python investigation chain. No real HIS/LIS records, LLM investigation, semantic grounding, hospital deployment, physician signature, or medical diagnosis is implemented.

M1-B remains the required first gate: every initial investigation **must have an explicitly confirmed Task Contract**, including a doctor-readable high-level outline. M2-A never overrides, auto-approves or bypasses that gate. The future full M2 must connect the approved Contract/actor/patient identity to C++ business persistence without trusting user-controlled identifiers.

## A. Runnable Python P001 slice (fake persistence)

The bounded Graph is:

```text
M1: doctor confirms Contract -> M1 graph END (status=contract_confirmed)
   -> M2: check M1 confirmed Run and revision
   -> prepare: synthetic get_patient_timeline
   -> persist_evidence: read E-002 original record (250mL);
                        read E-003 referral summary (about 800mL)
   -> close_issue: I-04 closed_unresolved (retain both sources)
   -> coverage_and_draft: six fixed template Sections + source-ID checks
   -> restricted_draft v1 -> M2 Graph END
```

The Graph State stores **only run_id, contract revision, phase and artifact references**; there is no doctor/patient ID or note content in its Checkpoint. Clinical scope and a confirmed Contract are checked by the M1 synthetic store **before and during each step**. `SyntheticP001Tools` only accepts DOC-001/P001, explicit known tool names, E-002/E-003 and records available at the synthetic cutoff. It rejects future/unapproved sources and never calls a prenatal model or invents a probability. No actual medical semantic verification is performed: the report clearly says `template_only=true` and `validation=template_source_ids_only_not_semantic_grounding`. This **must not** be marketed as an assertion-level grounded report.

Use a fresh synthetic-only directory:

```bash
python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
export PYTHONPATH=PythonServices/TreeSemAgent
DATA_DIR=/tmp/treesem-v2-m2-demo
python -m v2_m1.cli --data-dir "$DATA_DIR" start
python -m v2_m1.cli --data-dir "$DATA_DIR" confirm RUN_ID \
  --decision confirm --revision 0 --contract-revision 1
python -m v2_m2.cli --data-dir "$DATA_DIR" RUN_ID
python -m v2_m2.cli --data-dir "$DATA_DIR" RUN_ID
```

The final two calls return the same Draft v1 and the same four stored artifacts (two Evidence records, one IssueResult, one Draft), without duplicate operations. Neither doctor/patient ID is a credential. Revisions edited/clarified in M1 must be confirmed at their **new** revision before M2 is allowed to run.

SQLite `m2_artifacts.sqlite3` stores content in a separate business database; `m2_graph_checkpoints.sqlite3` is the official LangGraph Checkpointer. In `SyntheticArtifactStore.save`, each phase has a stable operation ID, a canonical request fingerprint, and an immutable artifact key. Insert and operation ledger are written under one `BEGIN IMMEDIATE` transaction. If an operation commits but its Graph node does not checkpoint, retry returns the prior artifact; a different payload under the same phase/op_id raises a conflict. **Graph nodes remain at-least-once, not exactly-once**. M2 Run-start orphan reconciliation and multi-worker shared thread ownership are not solved here.

Run automated tests:

```bash
PYTHONPATH=PythonServices/TreeSemAgent python -m unittest discover \
  -s PythonServices/TreeSemAgent/v2_m2/spec_tests -p 'test_*.py' -v
```

## B. Separate opt-in native C++/MySQL SIM Artifact API

`db/migrations/006_v2_sim_artifacts.sql` adds isolated `treesem_v2_runs`, `treesem_v2_artifacts`, and `treesem_v2_operations` tables with doctor/patient owners, confirmed state, unique artifact keys, source snapshots and operation fingerprints. It **does not modify V1 session/prediction/chat records**. Apply migrations normally, then build TreeSem with `KAMA_ENABLE_MYSQL=ON`, `KAMA_ENABLE_AUTH=ON`, `KAMA_ENABLE_ONNXRUNTIME=OFF` as desired.

The native API is **disabled by default**. To register its four prototype routes, explicitly set `TREESEM_V2_SIM_ARTIFACT_API=1` and use existing JWT authentication + MySQL storage; startup rejects this flag if auth/storage settings are incompatible. Routes accept **doctor Access JWT only** (not a caller-supplied doctor header), recheck doctor status and active patient assignment, bind the Run to that doctor/patient, and write to the database scheduler, not the Muduo I/O thread. No request can create/retrieve artifacts outside its owner scope.

```text
POST /api/v2/doctor/patients/:patient_id/sim-runs
     {"idempotency_key":"m2-example-01",
      "summary":"synthetic PPH records investigation",
      "overview":"review existing sources, then summarize limitations"}
     -> run_id, awaiting_confirmation, contract_revision=1

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/confirm
     {"expected_revision":0}
     -> contract_confirmed

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/artifacts
     {"operation_id":"op_E002_v1", "kind":"evidence",
      "artifact_key":"E-002:v1",
      "payload":{"classification":"SIM","source_id":"E-002",...}}
     -> immutable inserted/replayed result

GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/artifacts/:kind/:artifact_key
     -> stored JSON, owner checked
```

Prototype only: any uploaded payload must explicitly be `classification=SIM`. The API is not a trusted HIS/LIS source adapter, does not verify supplied clinical claims, does **not** accept real patient notes, and must never be enabled on a production clinical service. In MySQL, all artifacts are isolated per Run. Writes require confirmed state; Issue requires at least one Evidence, Draft at least one Issue. After Draft, new writes are blocked (same operation replay remains safe). Each write locks its Run row, checks source business state, inserts immutable artifact and operation result in **one transaction**; different payload under the same ID conflicts.

### Not yet wired / next changes before M2 complete

1. Use one **server-authenticated** Run/Contract store for M1 confirmation and subsequent M2 Tool/Artifact operations; do not pretend independent fake Python and C++ SIM Run IDs are a shared clinical identity.
2. Add a **scoped internal Python→C++ capability** authorized per tool/Run/patient, or an equivalent server-owned execution channel. Doctor JWT must **not** be silently reused as a long-lived Agent service credential. The current C++ public prototype APIs exist only for synthetic manually authenticated tests.
3. Add MySQL-backed integration tests with a test database and disabled fake patient data in public deployment; test cross-patient/cross-report isolation, idempotency retry, concurrent mutation and rejection of unsigned/unconfirmed Runs.
4. Replace template-only source-ID checks with M5 sentence/Assertion Grounding before making any clinical reliability claim; and add real dynamic Agenda only in M3.

Build and CI validation must be checked on the specific latest commit. When run-time MySQL integration is not executed, describe the native code as **compiled**, not end-to-end verified. **No main merge without approval.**
