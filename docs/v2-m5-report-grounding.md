# TreeSem V2 M5｜Report Coverage → Structured Writer → per-Assertion Grounding → Doctor revisions

**Status:** P001 SIM engineering vertical slice, with a deterministic Fake Writer and a separately configurable local OpenAI-compatible paragraph Writer. *No real Qwen run, no clinically validated natural-language entailment, no hospital HIS/LIS integration, no signed report.* This milestone tests engineering guardrails and provenance over a small **fixed approved assertion catalog**, not general medical truth.

## 1. Why M3's draft is not the M5 report

M3 generated an **internal historical T1 six-section template**, stored as a `draft` Artifact `report_{run_id}:v1`. Its paragraph text and candidate refs were not independently checked for sentence/Assertion support and it only covered E-001..E-003. M4 later loaded the **same LIS logical record** E-004/v1 Hb 110 → E-005/v2 Hb 130 and separately invalidated historical E-004/S-04 while creating current E-005/S-05. The old M3 historical template is **not automatically stale**, because it never cited E-004, but it is also **not the current evidence-supported report**.

M5 creates a **different, explicit report identity** `m5_{run_id}`, with immutable `draft_revision=1`; this avoids reinterpreting M3's earlier `report_{run_id}:v1` as already Grounded. The doctor must select `run_id + report_id + base_revision`. Same patient, different Run/report cannot be conflated by an LLM or inferred from chat.

## 2. A single factual example: multiple Assertions in one sentence

One of six paragraphs (P-02) has a single sentence containing **three distinct Assertions**:

- A-02-01-01: original delivery record's estimated blood loss **250 mL**, sourced to E-002/v1
- A-02-01-02: an independent **referral summary approximately 800 mL**, sourced to E-003/v1
- A-02-01-03: the two figures differ and remain unresolved, sourced to E-002/v1 **and** E-003/v1

The Writer provides *paragraph text and candidate evidence refs*, not authoritative citations. The deterministic reviewer splits each sentence, assigns stable sentence IDs/Unicode spans, enumerates every approved `fact_id`, and **independently fetches** Run-owned Evidence/IssueResult/M4 source versions to bind per-Assertion references. If even one text span is unrecognized or unsupported, the whole candidate is refused. Neither `confirmed PPH` nor `actual blood loss 800mL` can be passed by attaching a plausible source ID.

The other paragraphs cover: P-01 P001 self-report plus **current corrected Hb 130 g/L**; P-03 no unverified prenatal score; P-04 explicit **lack of reviewed literature**; P-05 old 110 g/L as **historical stale**, never current, plus limits on confirming PPH; P-06 restricted and unsigned uncertainty summary.

**Safety tradeoff:** The synthetic reviewer requires exact vetted full/concise textual variants in a 10-item assertion catalog. This supports deterministic end-to-end tests and the C++ server can repeat crucial checks. It is **not semantic Grounding for freely generated language**: a real Writer may paraphrase, in which case this M5 prototype will reject its text. A future semantically capable Reviewer and broader evaluation must be developed, measured and reviewed before general clinical use. The optional LLM Writer adapter exists but is **not** exercised by CI and has no reported success percentage.

## 3. Report Coverage and admission

`v2_m5/coverage.py` reads authoritative C++/MySQL Run and Contract revision; M3's E-001–003 and immutable I-01–04 IssueResults; M4's official version overlay/S-04 stale/S-05 current and pending I-05. It also queries M4's **current-reference** gate for E-001/002/003/005 and checks E-004 is marked historical stale. Missing or outdated inputs block publishing.

Coverage is **restricted**, not `complete`: I-02/I-04 source discrepancy unresolved; I-03 model prenatal applicability blocked; I-05 source correction needs_review; hospital guidelines/external literature not retrieved. Its objective mapping is a fixed **P001 test policy**, **not a semantic interpretation of arbitrary doctor Contract objectives**. Coverage checks presence/limitations, not clinical correctness. The report remains `clinical_unverified=true`, `not_signed=true`, `synthetic_catalog_only=true` with no risk score.

The `v2_m5/service.py` graph is

```text
M1 DoctorConfirm (must complete) → M3 Investigation (IssueResults)
                  → M4 official correction & versions
                  → M5 [Coverage] → [Compose + per-Assertion Review + Publish] → END
                                                ↓
                   C++ MySQL: immutable m5_report_v1
                                                ↓
                                 deliver restricted Draft to doctor
```

There is **no second mandatory doctor-review interrupt**. After initial delivery, doctors interact through a **separate report-specific pathway**, not by rerunning the M3 Investigation Graph.

## 4. MySQL versioning and doctor dialogue

New additive migration **010** creates:

- `treesem_v2_m5_reports`: one report bound to its generating Run/doctor/patient via the authoritative Run; current revision & Contract revision.
- `treesem_v2_m5_report_revisions`: immutable Draft v1/v2/v3 JSON snapshots with integrity digest.
- `treesem_v2_m5_operations`: unique operation IDs, complete payload fingerprint and saved responses.

The C++ server independently checks the P001 text/span/fact ID/source IDs/source version namespace and M4 current-source validity **in the same InnoDB transaction as the revision write**. A compromised caller cannot mark a free-form medical assertion as `supported` merely by setting a client boolean. The initial report `POST /internal/v2/sim-runs/:run_id/artifacts/m5/publish` requires a short-lived **Run/Tool capability**; a doctor Access JWT alone does not provide the internal publishing right.

Doctors use two explicit public routes:

```http
GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m5/reports/:report_id/revisions/:revision

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m5/reports/:report_id/messages
{"conversation_id":"cv_001","intent":"explain","base_revision":1,"paragraph_id":"P-02"}

POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m5/reports/:report_id/messages
{"conversation_id":"cv_001","intent":"shorten","base_revision":1,
 "paragraph_id":"P-02","operation_id":"op_...","document":{...complete reviewed Draft v2...}}
```

- **Explain** is read-only. It returns the actual selected paragraph and its source-bound Assertions, not an invented new RAG answer, Tool call or silently regenerated Draft.
- **Shorten** is a deliberately bounded P001 demonstration: switch one paragraph from an approved full phrase to a strictly shorter **same-facts** variant. Re-run Review for **all six paragraphs** including adjacent citations. The entire proposed Draft v2 must equal the previous version except the chosen paragraph plus revision/Writer-kind metadata. MySQL uses **current_revision CAS**, and only then stores an immutable new version and operation record. Another edit with the same base but a different operation ID conflicts, while retrying an identical accepted operation returns its original result. Two sequential shortening turns produce v2 and v3, while explanation may occur at any selected current revision.
- Doctor identity, active doctor–patient assignment, selected report ID, generating Run, selected paragraph, Contract revision and current source versions are checked at read/publish/edit. Clients **must not infer report identity** from a conversation ID, disease name, or patient match.

The pipeline obtains a new short-lived capability for doctor-driven source validation as needed, rather than putting doctor Access JWT, model prompts, raw medical records or capability tokens into LangGraph State/SQLite Saver.

## 5. Checkpoint/DB crash window

Graph checkpoints contain only Run ID, Contract revision, phase and report reference. The deterministic Fake Writer/Reviewer recomputes the same reviewed report if **MySQL publish COMMIT succeeded but Graph did not checkpoint**; a stable initial operation ID replays the saved SQL result and does not generate a second report. Doctor edits use a unique operation ID derived from explicit request identity and base version, with a hash of the *complete proposed payload*.

**Unfinished production concern:** an optional real local LLM Writer can be nondeterministic. If SQL commits but the graph node re-executes the LLM and creates different accepted wording, the stable op ID conflicts **safely**, but resuming uninterrupted requires a durable pre-publish Writer proposal/outbox. That, multi-worker Checkpointer/fencing and general report-change recovery are **M6**, not claimed here as solved.

## 6. Actual acceptance tests

Run Python fake Reviewer tests:

```bash
python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
PYTHONPATH=PythonServices/TreeSemAgent python -m unittest discover \
  -s PythonServices/TreeSemAgent/v2_m5/spec_tests -p 'test_*.py' -v
```

GitHub Actions `v2_m5_mysql_integration` starts MySQL 8, applies migrations 001–010, compiles/starts the real JWT-authenticated C++ gateway, and executes:

- M1 confirmed Contract → M3 dynamic Investigation → M4 EV-001 correction → M5 restricted Draft v1 without a second doctor gate.
- Six sections, ten separately proved Assertions, correct Unicode start/end spans, source versions (E-004 **historical stale**, E-005 **current**), absent risk score.
- Doctor `explain(P-02)` without a new revision, then `shorten(P-02)` to v2, then `shorten(P-05)` to v3; all other paragraphs and evidence bindings remain unchanged.
- No new version on duplicated operation; stale CAS rejected; forged/superseded or unrelated report ID rejected; doctor JWT cannot publish internally; scoped Agent token cannot impersonate doctor.
- Fault injection pre-commits v1 before Graph execution on a second separate Run, then replays; report and operation counts remain 1 each. Doctor Assignment revocation denies subsequent read/explanation.
- Existing M0–M4/V1 suites remain independent. Refer to the **actual passing CI run at current HEAD** before marking this complete.

**Non-goals**: real clinical free-text NLI, validated model or physician recommendation, actual trusted HIS/EMR/LIS/knowledge evidence, arbitrary report rewriting, disambiguating patient identity from chat, formal signing, PHI retention/deletion, clinical certification, and generalized per-sentence scoring percentages. Doctor editing is intentionally minimal; broader request classification, targeted follow-up investigation and cross-document reasoning remain future work.
