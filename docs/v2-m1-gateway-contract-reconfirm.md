# M1 Gateway 回补｜MySQL Task Contract 版本化 / 澄清 / 再确认

> Work order: **M3 implemented and verified first → return to the missing M1 Gateway feature**, before M4. This is a follow-up to M1-B's original SQLite-only revise/clarify behavior, **not a redesign of M3**.

## Product flow (same mandatory HITL node)

```text
C++ JWT doctor + authorized patient → MySQL create Run, Contract v1
        ↓
M1 Graph contract_gate interrupt (shown goal, overview, clarification)
        ├─ clarify: answer pending question → atomic MySQL Contract v2
        │         → LangGraph re-enters contract_gate interrupt
        ├─ revise: edited goal + high-level outline → Contract vN+1
        │         → LangGraph re-enters same contract_gate interrupt
        ├─ cancel: terminal cancelled → END (no capability / investigation)
        └─ confirm: only with NO unanswered clarification
                  → MySQL status=contract_confirmed → Graph END
        ↓
M2 or M3 starts using the EXACT confirmed run_id and contract_revision
```

Even a seemingly clear initial request **must** pass DoctorConfirm. The optional synthetic clarification question only changes whether the doctor can immediately choose confirm; if a clarification is unanswered, confirm is rejected **before** the response is submitted to LangGraph. The initial overview describes *scope/direction*, not a locked Tool itinerary; M3's Agenda continues to be dynamic. No Report review interrupt follows Draft delivery.

## Authoritative persistence and protected operations

Migration `008_v2_contract_versions.sql` is additive after 006/007 and creates `treesem_v2_contract_versions` (run_id + contract_revision immutable key, summary, investigation_overview, optional pending question + answer, created_at). It backfills existing M2 SIM Runs with their original Contract v1, and permits a terminal `cancelled` Run status. **Migration files 006/007 remain unchanged**, so an existing M2-B database can be upgraded rather than destructively recreated.

Each doctor action is an authenticated C++ MySQL transaction that:

1. Revalidates doctor identity and active patient Assignment, then locks its exact Run row.
2. Computes stable operation ID **before** any database write: SHA256(run_id|contract|expected_business_revision), and fingerprints decision + contract_revision + normalized payload and owner context.
3. **First checks the committed Operation Journal**. Same operation and payload returns recorded result, even when the MySQL Run is already a later Contract revision or a Draft. Reusing the same ID with a different payload is a conflict.
4. For a fresh request, performs a status check, optimistic `business_revision` and `contract_revision` checks, and validates the pending clarification requirement. A new revise/clarify inserts immutable Contract vN+1, updates the Run's latest revision and summary, and commits its journal in **one** InnoDB transaction.
5. `confirm` changes status to contract_confirmed at the exact verified Contract revision (no new Contract version); `cancel` changes status to cancelled (no new Contract version). Both are journalled, replay-safe and cannot be silently undone.

The two distinct revision counters serve different purposes: business_revision increments on *every* doctor action; contract_revision increments **only** when a revised/clarified Contract snapshot is created. For example: Start (business0, Contract1) → Clarify (business1, Contract2) → Revise (business2, Contract3) → Confirm (business3, Contract3). A stale original Contract1 confirmation must fail. Unlimited clarification loops are not permitted in this synthetic prototype (maximum Contract revision 10); cancellation remains possible.

### API (doctor JWT, not the Agent capability)

```http
POST /api/v2/doctor/patients/:patient_id/sim-runs
{"fixture_id":"P001","idempotency_key":"...","summary":"...",
 "overview":"...","clarification_question":"optional nonempty question"}

GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id
 -> business revision, Contract revision, current summary/overview,
    clarification_question, clarification_answer

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/contract
{"decision":"clarify","expected_revision":0,"contract_revision":1,
 "answer":"允许注明来源限制但不做未经验证的诊断"}

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/contract
{"decision":"revise","expected_revision":1,"contract_revision":2,
 "summary":"仅核查有冲突的来源","overview":"按证据动态增加调查问题"}

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/confirm
{"expected_revision":2,"contract_revision":3}
```

`GatewayDoctorClient` handles these signed doctor-only calls. `GatewayContractWorkflow` invokes the original M1 `build_graph()` and handles `revise/clarify/cancel/confirm` by using the native transactions rather than storing a second business Run in Python SQLite. Rejected responses are validated before `Command(resume)` to avoid poisoning a paused LangGraph task; a transaction-level authority check remains mandatory after preflight.

## Crash-window example and exact verification

Crash case: MySQL has already committed revised Contract v2 + operation result, but LangGraph SqliteSaver still contains a pending node for Contract v1 and Run business revision 0. On restart the same **doctor action** is submitted to the same thread. The C++ store checks operation ID and fingerprint **before current-version mismatch checks**, returns the exact prior v2 result, and Graph resumes to the next interrupt. No Contract v3 is created on replay.

The `tests/test_v2_m1_gateway_integration.py` regression starts the actual C++ binary with MySQL 8 and fake accounts/doctor-patient assignments, uses real M1 interrupt, simulates invalid confirm → valid clarify → revise → explicit re-confirm; checks SQL Contract history and operation counts; simulates both edit and confirm business COMMIT before Graph Checkpoint, tests same-ID different-payload conflict, cancel denial, stale revision rejection, wrong patient owner, and runs the **M3 Agent on a Run whose confirmed Contract is v3**. This proves that backfilling the M1 pre-Investigation edit feature does **not** require rewriting M3.

The old M2/M3 integration jobs continue running. A passing C++ compile alone is insufficient; cite a **completed/success** `v2_m1_gateway_integration` run before claiming the whole native HITL cycle is verified.

## Boundaries (do not overclaim)

- Still **SIM P001 only**. A doctor-entered optional question and revised text are not LLM-generated semantic clarification. The UI/API passes typed structure. A real Contract parsing/evaluation component remains outside this task.
- No real HIS/EMR/LIS medical records, production clinical security certification, external-provider model inference, or verified medical Writer/Grounding.
- The approved Contract is the **WHAT** constraint for M2/M3; the Agent still chooses **HOW** within Guard. If the user modifies an already-confirmed report later, that is the separate Report Revision workflow (M5), not re-opening this Contract for silent overwrite.
- Graph Checkpointer and MySQL are separate databases and do not share one transaction; operation idempotency plus checkpoint re-entry bound the crash window. Multi-worker fencing/orphan repair and durable encryption remain later concerns.
- No production-ready browser UI or human medical signature; the tests use fake doctor access tokens and fake records.

This follow-up is staged as a new Draft PR on M3 PR #5 and is **not merged to main**.
