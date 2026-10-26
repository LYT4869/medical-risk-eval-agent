# TreeSem V2 M8 — Fake-first dynamic Investigation Runtime

## Scope and evidence

Implemented opt-in P001 synthetic (SIM) Agent runtime; no real Qwen, clinical NLI, hospital HIS/EMR/LIS, generic reporting or C++ MySQL integration. Reconciled against Notion 02 (ADR-009/011/013), 03 (Agenda / Context / IssueResult contract), 07 (Q05–Q08 / Q14), P001 T02/T04/T06. Notion 08–09 are not the basis for design verification.

This adds a runnable LangGraph path in PythonServices/TreeSemAgent/v2_m8, not a replacement for the existing M3 → M5 → M6 fixed catalog pathway.

## Key files and behavior

- contracts.py: shared model-proposal contracts for plan_task / decide / consolidate. Initial Agenda now comes from an injected Agent proposal rather than fixed Runtime topics. Runtime owns stable Issue IDs, objective coverage, revision CAS, exact-question duplicate detection, source ID allowlists, capacity and state constraints. No automatic semantic-equivalence or diagnosis validation.
- context.py: Run-scoped get_evidence and get_issue_result; context_for_issue loads related completed Result first, then unresolved, then limited other history. Maximum 4 historical results and 5 evidence projections. No broad patient search, raw PHI, or conversation-history dump.
- close.py: callable close_issue workflow, entered by an Agent action. It loads context, calls the SAME injected Agent consolidator, checks candidate Statement structure/references, validates follow-up Delta before commit, and writes immutable IssueResult + approved Agenda Delta in one artifact. On Graph retry, previously committed artifact is read without regenerating proposal. No second Planner/Memory Agent.
- fake_agent.py: scripted semantic decisions on the synthetic P001 source conflict; all domain-dependent rules live in the Fake substitute rather than being built into general Runtime validation. A different single-objective plan is supported.
- workflow.py: independent LangGraph seed → decide → execute iteration; first proposed Agenda persisted as immutable agenda_seed, scoped read-only P001 sources, Close candidate durability, budget and no-progress block. Invoked via M8LocalWorkflow only after M1 SIM mandatory doctor-confirmed Contract. Uses a separate local SQLite Artifact store and official SqliteSaver.
- v2_m3/repository.py: extends LOCAL M3 store kind allowlist for agenda_seed only. Existing scoped C++/MySQL Gateway and M3 artifact contract are unchanged.
- spec_tests/test_runtime.py: model-independent regression including dynamic initial plan, Delta CAS and duplicate/invalid rejection, cross-Issue Result reuse, unauthorized source, doctor-confirmation gate, invalid model proposals, alternate plan, checkpoint replay, SQL committed before graph checkpoint recovery, and candidate semantic status.

## Run locally

    python -m pip install -r PythonServices/TreeSemAgent/requirements-v2-m1.txt
    export PYTHONPATH=PythonServices/TreeSemAgent
    python -m unittest discover -s PythonServices/TreeSemAgent/v2_m8/spec_tests -p 'test_*.py' -v

CI job: v2_m8_dynamic_runtime. Existing M3/M5/M6 tests must stay green.

## Important limitations

1. These are ENGINEERING acceptance tests using deterministic Fake proposals; success does not show that any Qwen model can discover correct new Issues, consolidate medical facts, or remember correctly.
2. New IssueResult statements are marked semantic_validation=not_evaluated, status=closed_unverified. They are NOT safe to publish as verified medical assertions. A true semantic reviewer and doctor review are separate future gates.
3. Only exact normalized question duplicate detection and trusted ID/revision validation are implemented. Semantic de-dup, search_evidence, generic cross-Issue RAG, revalidation of all stale statements and production concurrency remain out of scope.
4. Existing M5/M6 C++ Validator explicitly expects fixed P001 facts and Issue IDs. Do not feed M8 unverified general Result into report Publish or weaken existing Guard. Integrating dynamic semantics with authoritative MySQL/Report is a separate next milestone.
5. The real-model adapter for the new plan_task/decide/consolidate contract is not connected yet. Billing credentials, Qwen results, token and latency metrics remain absent.
