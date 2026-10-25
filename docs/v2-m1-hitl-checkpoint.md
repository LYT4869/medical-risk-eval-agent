# TreeSem V2 M1-B: mandatory Task Contract confirmation (synthetic demo)

M1-B implements the **first mandatory DoctorConfirm stage** in an isolated LangGraph demo: preview the synthetic Task Contract, `interrupt()` until the doctor confirms, and keep clarification/revision **inside the same confirmation phase**. A revision or answered clarification creates an immutable draft version, then returns to `contract_gate` for explicit reconfirmation. A confirmed contract reaches `contract_confirmed → END` **without a report review gate**.

**What is NOT implemented:** there is no real LLM Contract parser, no real doctor authentication, no C++ gateway, HIS/LIS, Evidence/IssueResult persistence, Investigation/Agenda, Writer, Grounding, report, Draft delivery or clinical signoff. The high-level outline is a static synthetic preview, **not a validated investigation plan**. `contract_confirmed` only marks a future M2 investigation boundary, **not a completed report**. M1-A's earlier second `review_gate` was an interrupt/replay experiment; it is removed from the M1-B product-shaped graph.

## Reproduce locally

Python 3.10+; run from repository root:

```bash
python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
cd PythonServices/TreeSemAgent
python -m unittest discover -s v2_m1/spec_tests -p 'test_*.py' -v
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b start
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b status RUN_ID
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b confirm RUN_ID --decision confirm --revision 0 --contract-revision 1
```

Use the `run_id` returned from `start`, and the exact `revision` and `contract.revision` shown by `status`. **Always explicitly confirm even for a clear synthetic request.** Each CLI invocation is a distinct Python process sharing SQLite files.

A doctor-changing-the-goal example (new data directory / Run ID; never re-use old revision numbers):

```bash
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b start
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b confirm RUN_ID --decision revise --revision 0 --contract-revision 1 --summary "仅核查已有来源差异" --overview "先核对原始病历，再按证据动态补查"
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b status RUN_ID
python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1b confirm RUN_ID --decision confirm --revision 1 --contract-revision 2
```

To exercise clarification, `start --ask-clarification` creates one **synthetic** open question, and `confirm --decision clarify --revision 0 --contract-revision 1 --answer "允许受限草稿但必须注明依据不足"` stores the answer in a new Contract revision. It **still waits** for a separate `confirm --decision confirm --revision 1 --contract-revision 2` action. An unanswered question blocks confirmation; `cancel` is terminal. This small deterministic fixture does not produce a revised LLM-generated semantic summary when answering; that is deferred.

**Incompatible with old M1-A demo database schema:** use a **new** `--data-dir` to create fresh synthetic SQLite files; `setup()` deliberately rejects the prior operation table format. Never feed patient-identifying data to this CLI. DOC-001/P001 are hardcoded synthetic IDs, **not credentials or an authentication system**.

## State, transaction boundaries, crash recovery

```text
START
  -> contract_gate: interrupt(Contract ID + business/contract revisions)
      | revise/clarify => atomic Contract vN+1 + operation journal
      |                   -> contract_gate: interrupt again
      | cancel         => cancelled -> END
      | confirm        => contract_confirmed -> END
```

- `RunState` has only run_id, contract_revision, business_revision, phase and the next pre-derived operation ID. Patient/actor IDs, full medical facts, tokens and Contract summary are **not** placed into Graph State. `SqliteSaver` checkpoints the graph at stable `thread_id=treesem-v2:<run_id>`, with `durability="sync"` for this local single-worker demo.
- `DemoRunStore` uses **different SQLite storage** from LangGraph: `v2_demo_runs` (business version/status), `v2_demo_contracts` (immutable Contract drafts), and `v2_demo_operations` (unique logical operation ledger + request SHA256 fingerprint + persisted result status/business/contract revisions).
- Every commit uses `BEGIN IMMEDIATE`, validates actor/run, contract and business revision, persists the new immutable draft (for revise/clarify), updates Run by compare-and-set, and saves the operation journal **in one DB transaction**. The operation ID uses `run_id|contract|expected_business_revision`, derived **before** effects.
- If business COMMIT succeeds and the Graph checkpoint has not, the node may replay with its **old contract_revision**. We must check the immutable operation journal **before** checking the business Run's current contract version; identical request bytes/payload return the *old* recorded result rather than inserting another Contract draft. A different payload under the same ID conflicts. The next graph node resumes only after its state update is checkpointed.
- This is **idempotent business effect**, not exactly-once node execution or multi-database atomicity. Graph restart works on the same thread ID; it does not magically migrate actor permissions or patient data. Reauthorization, cross-worker concurrency, reconciliation for orphan Run creation, and production-grade encryption/retention are deferred.

## Verification and next integration

The isolated M1 CI job installs pinned LangGraph/Saver, runs the business-store tests, real Graph tests and real separate-Python-process CLI tests; V1 tests run in the independent `fast` job. Passing M1-B would verify only **synthetic Contract workflow and SQLite replay**, not safe clinical operation or actual Draft delivery.

Next (M2): connect authenticated C++/MySQL Run/Artifact API, allow the first confirmed Contract to start a bounded synthetic Investigation/Tool→Evidence→IssueResult sequence, and only then persist and **immediately deliver** a clearly marked Draft v1 after minimum Coverage/Grounding checks. No second mandatory DoctorReview interrupt. Formal publishing/signature, if implemented later, needs a separately authorized audited business action.
