# TreeSem V2 M2 — Synthetic vertical slice, M2-A and M2-B

> **SIM-only implementation**. This is an engineering prototype, NOT an EHR interface, validated prenatal inference, a medically grounded clinical report, or a physician signature. The sample fixture is P001 and cannot be silently substituted with real patient notes.

## Current architecture

M2-A's independent Python/SQLite fake remains as an offline regression path. **M2-B connects a separate gateway variant with a single authoritative C++/MySQL Run**, reusing the *actual* M1-B LangGraph `contract_gate` interrupt:

```txt
[Authenticated doctor access JWT]
    -> C++ Gateway: CREATE V2 SIM Run + Contract preview in MySQL
    -> Python M1 LangGraph: contract_gate interrupt (graph-only SQLite checkpoint)
    -> Doctor confirms via authenticated request
    -> M1 resumed node: C++ confirm(run_id, expected_revision=0)
    -> MySQL contract_confirmed  (business truth)
    -> C++ issues short-lived Run+patient+doctor scoped tool JWT
    -> Python M2 Graph (new Checkpointer, same run_id):
          get_patient_timeline [fixed synthetic adapter, E-002/E-003]
          read_patient_record   [fixed synthetic adapter]
          ↓
          HTTP/internal/v2 -> C++ Owner/Assignment/Run/Auth Guard
          ↓
          MySQL ArtifactRepository: Evidence E-002/E-003
          ↓
          MySQL I-04 IssueResult (unresolved conflict)
          ↓
          six-section, restricted SIM template Draft v1
    -> C++ MySQL: persist immutable Draft + journal
    -> M2 Graph END (no DoctorReview interrupt)
    -> Doctor GET selected report_id/revision (doctor's authenticated route)
```

**One authoritative business Run**, not two: new `GatewayContractWorkflow` and `GatewayM2Workflow` use the C++ Run created by the server, while their SQLite files are **only LangGraph Checkpoints**, not duplicate Run/Artifact tables. The original `v2_m1.DemoWorkflowService` and `M2SyntheticService` intentionally remain standalone local technical demonstrations. The gateway path currently supports **initial Contract confirm only**; the M1-B offline `revise/clarify` loop is **not yet supported by the C++ Contract API**. Never describe M2-B as having wired the entire M1 editing path.

### Doctor and Agent API boundaries

The V2 SIM API is **off by default**. It requires the explicit `TREESEM_V2_SIM_ARTIFACT_API=1` flag, `TREESEM_AUTH_MODE=required`, MySQL-enabled build and MySQL schema migration 006. All writes must declare `classification: "SIM"`, and a new Run must explicitly select `fixture_id: "P001"`. The `patient_id` URL identifies an authenticated, doctor-assigned **test patient account**, NOT the identity of a real P001 record.

Public authenticated **doctor Access JWT** endpoints:

```http
POST /api/v2/doctor/patients/:patient_id/sim-runs
{"idempotency_key":"unique-demo-request-id",
 "fixture_id":"P001",
 "summary":"what doctor wants to inspect",
 "overview":"short high-level direction for doctor confirmation"}

GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id
POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/confirm
{"expected_revision":0}

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/capability
{}
GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/artifacts/:kind/:artifact_key
```

**Only after a confirmed Contract** does C++ sign a short-lived `treesem-internal-tools` JWT capability scoped to a trusted doctor, assigned patient, exact run_id, and `v2_sim_status / v2_sim_artifact_read / v2_sim_artifact_write` tools. It is not a doctor access token, not a service-wide secret, and is **never put in Graph State/Checkpoint or report JSON**. The Python adapter carries this transient token **in memory** when calling:

```http
GET  /internal/v2/sim-runs/:run_id
POST /internal/v2/sim-runs/:run_id/artifacts
{"operation_id":"op_...",
 "kind":"evidence|issue|draft",
 "artifact_key":"stable source/version or issue/draft key",
 "payload":{"classification":"SIM", ...}}
GET  /internal/v2/sim-runs/:run_id/artifacts/:kind/:artifact_key

Authorization: Bearer <scoped capability>
X-TreeSem-Session-Id: v2:<run_id>
```

The C++ `SecurityMiddleware` verifies JWT audience/signature/expiration/tool scope and replaces any caller-supplied actor/role/subject/run headers; `V2SimController` further requires the route's run_id == signed run claim, doctor role+active user, and calls `AuthService.authorizeSubject` on **every** read/write (including after assignment revocation). `V2SimArtifactStore` checks that doctor/patient exactly own the Run and that an explicit confirmation happened. Public doctor requests **cannot write arbitrary Artifacts**. The capability endpoint returns the short-lived token only to the authorized doctor application; production orchestration must define an appropriately protected handoff to the Agent.

### Business transaction and replay behavior

`db/migrations/006_v2_sim_artifacts.sql` introduces `treesem_v2_runs`, `treesem_v2_artifacts`, and `treesem_v2_operations`; V1 tables are not modified. Creation is idempotent by (doctor, patient, idempotency_key, contract fingerprint). All Artifact writes happen under a transaction with an InnoDB Run row lock; the same transaction inserts the immutable payload and its operation journal. An unchanged request with the same `operation_id` returns its recorded result, even when the Run's status became `draft_ready`. Reusing the ID with a different payload or attempting to overwrite an immutable source returns a conflict; a report must match its Run and remain unsigned, template-only and restricted.

This is **exactly-once business effect for a stable operation ID**, not exactly-once Graph execution or a distributed transaction. A C++ commit can happen before the Python Graph checkpoint; the next Graph attempt should safely replay the committed write. Graph operations and tool calls are still at-least-once. A recheck of doctor/subject/Run authorization is required even on a replay. Backend/Agent thread ownership, general reconciliation and PHI retention are not production-ready.

### Reproduce and verify

To validate locally, use an isolated MySQL 8 **test database**, with migrations 001–006 applied and the server compiled using MySQL/JWT:

```bash
TREESEM_DB_PASSWORD='your-local-test-password' \
TREESEM_DB_HOST=127.0.0.1 TREESEM_DB_PORT=3306 \
TREESEM_DB_NAME=treesem_test TREESEM_DB_USER=treesem_app \
bash scripts/migrate_treesem_db.sh

cmake -S . -B build-v2-integration \
  -DKAMA_ENABLE_ONNXRUNTIME=OFF -DKAMA_ENABLE_MYSQL=ON \
  -DKAMA_ENABLE_AUTH=ON -DKAMA_BUILD_TESTS=OFF
cmake --build build-v2-integration -j2

python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
PYTHONPATH=PythonServices/TreeSemAgent \
TREESEM_DB_HOST=127.0.0.1 TREESEM_DB_PORT=3306 \
TREESEM_DB_NAME=treesem_test TREESEM_DB_USER=treesem_app \
TREESEM_DB_PASSWORD='your-local-test-password' \
python tests/test_v2_m2_gateway_integration.py build-v2-integration/treesem_server
```

The integration test creates **fake** users/assignments, signs test-only JWTs, starts the actual C++ binary, checks M1's real interrupt+MySQL confirmation, obtains a C++ signed V2 capability, runs Python M2 with both E-002 and E-003, checks MySQL operation/artifact counts, reads Draft through public doctor API, restarts C++, verifies persistence, revokes the assignment, and tests invalid token, cross-Run, replay and unauthorized access boundaries.

CI job `v2_mysql_integration` runs an ephemeral MySQL 8 service alongside the actual C++/Python processes; independent M1, M2-A and existing V1 jobs run unchanged. **Only claim pass for the exact workflow run actually observed.**

### Deferred, even after M2-B

1. **Contract edits with the gateway**: original M1-B local revise/clarify works; native MySQL V2 Run currently supports explicit initial confirmation only. Add a versioned native Contract edit/reconfirm transaction + tests before saying that branch is integrated.
2. **Real data Tool adapters**: `get_patient_timeline` and `read_patient_record` still read the fixed P001 JSON fixture in Python. The C++ Gateway authenticates and persists artifacts, but does not source real HIS/EMR/LIS data; a trustworthy clinical Tool source is outside this simulation.
3. **Actual clinical Evidence verification**: template report validation is only source-ID and fixed-text checking. Sentence/Assertion Grounding, LLM ReAct/dynamic Agenda, source correction propagation, clinical review and formal signing are separate milestones.
4. **Production execution/security**: no browser-based doctor UI, no durable encrypted secret distribution, no server-initiated internal capability handoff, no PHI processing policy, no distributed multi-worker Checkpointer fencing, and no claim of validated medical risk prediction. Never point this SIM API at production patient records.

## Delivery history

- M0: immutable P001 synthetic chronology Fixture.
- M1-A: two-interrupt LangGraph technical recovery demo (historical).
- M1-B: single mandatory initial Contract HITL with offline revise/clarify loops; no mock DoctorReview.
- M2-A: P001 synthetic Tool→Evidence→IssueResult→Draft Graph, Fake SQLite repository, plus independent C++ SIM API compiled.
- **M2-B: native MySQL Run and scoped C++ Artifact transport wired into M1 first-confirm path + M2 Graph, with dedicated real MySQL integration regression.**

All contributions are staged through PR #4, stacked on M1 PR #3, and **not merged to main**.
