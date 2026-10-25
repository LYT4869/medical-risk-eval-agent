# TreeSem V2 M1: LangGraph HITL + checkpoint (synthetic demo)

**Implemented:** separate Python M1 demo with two real LangGraph interrupts and official persistent SQLite Saver. This does **not** modify the V1 AgentLoop, C++ gateway, ONNX, or MySQL. There is no clinical Tool, Evidence Store, report generation, or actual physician approval. The second interrupt is a **mock review acknowledgement**, not report signing.

## How to run

Python 3.10+; from repository root, install dependencies:

    python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt

Then run:

    cd PythonServices/TreeSemAgent
    python -m unittest discover -s v2_m1/spec_tests -p 'test_*.py' -v
    python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1 start
    python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1 status RUN_ID
    python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1 confirm RUN_ID --decision confirm --revision 0
    python -m v2_m1.cli --data-dir /tmp/treesem-v2-m1 review RUN_ID --decision acknowledge --revision 1

Replace RUN_ID with the start command's output. Each CLI command uses a **new process**. DOC-001 and P001 are synthetic identifiers only: the CLI does not authenticate real users and **must never become a clinical endpoint**.

## Architecture and checkpoints

- workflow.py: START -> contract_gate(interrupt) -> review_gate(interrupt) -> END. Contract cancel/revise ends at cancelled/needs_revision; real contract editing and reconfirmation are deferred. No LLM used.
- service.py: derived stable thread ID from run ID; official SqliteSaver stores Graph checkpoints in graph_checkpoints.sqlite3 with durability="sync". Each new process reconstructs the Graph, checks get_state, and issues Command(resume=...) only at the expected waiting gate.
- The Graph State contains only run ID, revision, phase, and **stable operation IDs**. No actor, patient, clinical text, credentials or Capability tokens.
- run_store.py and transitions.py simulate separate C++-owned business state in demo_business.sqlite3: business Run status and operation journal; SQLite BEGIN IMMEDIATE, compare-and-set revision, UNIQUE(run_id, phase, operation_id), SHA-256 request fingerprint and recorded operation result.
- The operation ID derives from **run ID + phase + expected revision** *before* the side-effecting node. Retry with the same payload returns the old result, changing the payload under the same ID raises conflict. This gives idempotent **business effects**, not exactly-once execution.
- Crash-window test: if the business transaction commits before Graph checkpoint, the Graph still waits at the old node. On retry, the durable operation journal returns the old result, so no duplicate business transition occurs.

## Production boundary and verification

This is a local SQLite demo. Real V2 will put Run, Evidence, IssueResult, Report and idempotent operations behind the **existing C++ business API and MySQL**, with real actor/patient/assignment reauthorization on resume. V1 short-lived JWT Capability tokens must not be checkpointed or prolonged to days. SQLite Saver is for a single-worker prototype; ADR-012 targets the official PostgreSQL Saver for multi-worker deployment, if permitted. Neither Saver can atomically commit with the business MySQL.

The new CI job runs store, Graph and **cross-process** tests with pinned LangGraph/SQLite Saver. This demonstrates state persistence and synthetic idempotency, **not medical reliability**. M2 will introduce MySQL-backed, authorized ArtifactRepository, HIS/LIS mock adapters and genuine Evidence write/replay tests.
