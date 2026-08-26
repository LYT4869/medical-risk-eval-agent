import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.llm_client import ScriptedLlmClient
from agent.loop import AgentExecutionError, AgentLoop
from agent.prompt import SYSTEM_PROMPT
from agent.schemas import AgentRunRequest, LlmToolCall, LlmTurn
from agent.skills import SkillCatalog
from agent.tool_registry import ToolRegistry

from test_agent_loop import FakeBackend


class FakeKnowledge:
    def __init__(self):
        self.calls = []

    async def search(self, token, query, scope, top_k, trace=None):
        self.calls.append((token, query, scope, top_k))
        return {
            "index_version": "knowledge-test",
            "retrieval_mode": "hybrid",
            "results": [{
                "citation_id": "cite_" + "a" * 20,
                "source_id": "src_who_pph",
                "title": "WHO PPH guideline",
                "section": "Recommendations",
                "page": 12,
                "excerpt": "Curated evidence excerpt.",
                "publisher": "World Health Organization",
                "published_at": "2025-01-01",
                "url": "https://www.who.int/example",
                "content_sha256": "b" * 64,
                "score": 0.9,
            }],
        }

    async def ready(self):
        return True

    async def close(self):
        return None


def run_request(role="patient"):
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message="请解释产后出血知识",
        actor_role=role,
        knowledge_capability_token="signed-knowledge-token")


class KnowledgeGroundingTest(unittest.TestCase):
    def test_registry_filters_definitions_to_guard_scope(self):
        registry = ToolRegistry(FakeBackend(), FakeKnowledge())

        names = {
            item["function"]["name"]
            for item in registry.definitions(
                allowed_tools={"search_medical_knowledge"})
        }

        self.assertEqual(names, {"search_medical_knowledge"})

    def test_tool_descriptions_define_minimum_routing_boundaries(self):
        registry = ToolRegistry(FakeBackend(), FakeKnowledge())
        descriptions = {
            item["function"]["name"]: item["function"]["description"]
            for item in registry.definitions()
        }
        self.assertIn("Use this alone", descriptions["get_explanation"])
        self.assertIn("Only use when", descriptions["get_prediction"])
        self.assertIn("Do not use for a stored prediction explanation",
                      descriptions["search_medical_knowledge"])
        self.assertIn("minimum sufficient tool set", SYSTEM_PROMPT)
        self.assertIn("do not repeat the search", SYSTEM_PROMPT)

    def test_system_prompt_refuses_explicit_abuse_without_tools(self):
        self.assertIn("refuse directly without calling any tool", SYSTEM_PROMPT)

    def test_search_and_citation_grounding(self):
        citation = "cite_" + "a" * 20
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH", "scope": "clinical", "top_k": 5})]),
            LlmTurn(content=f"Evidence is available [{citation}].",
                    grounding_source_ids=[citation]),
        ])
        knowledge = FakeKnowledge()
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), knowledge)).run(run_request()))
        self.assertEqual(result.grounding_source_ids, [citation])
        self.assertEqual(result.citations[0].source_id, "src_who_pph")
        self.assertEqual(result.knowledge_index_version, "knowledge-test")
        self.assertEqual(knowledge.calls[0][0], "signed-knowledge-token")

    def test_citation_metadata_is_derived_from_verified_answer_mentions(self):
        citation = "cite_" + "a" * 20
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH", "scope": "clinical", "top_k": 5})]),
            LlmTurn(content=f"Evidence is available [{citation}]."),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(run_request()))
        self.assertEqual(result.grounding_source_ids, [citation])
        self.assertEqual(result.citations[0].citation_id, citation)

    def test_verified_claimed_citation_is_appended_to_answer_when_omitted(self):
        citation = "cite_" + "a" * 20
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH", "scope": "clinical", "top_k": 5})]),
            LlmTurn(content="Evidence is available.",
                    grounding_source_ids=[citation]),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(run_request()))
        self.assertIn(citation, result.answer)
        self.assertEqual(result.grounding_source_ids, [citation])

    def test_missing_citation_has_internal_reason_without_public_leakage(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH", "scope": "clinical", "top_k": 5})]),
            LlmTurn(content="Evidence is available, but no citation was supplied."),
        ])

        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(run_request()))

        self.assertEqual(result.policy_rejection_code,
                         "missing_knowledge_citation")
        self.assertNotIn("policy_rejection_code", result.model_dump())
        self.assertIn("无法提供未经可信工具结果验证", result.answer)

    def test_fabricated_citation_returns_safe_fallback(self):
        llm = ScriptedLlmClient([LlmTurn(
            content="Unsupported [cite_" + "f" * 20 + "].",
            grounding_source_ids=["cite_" + "f" * 20])])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(
                run_request().model_copy(update={
                    "message": "帮我看看这个情况"})))
        self.assertIn("无法提供未经可信工具结果验证", result.answer)
        self.assertEqual(result.grounding_source_ids, [])

    def test_missing_knowledge_capability_is_safe_tool_error(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH"})]),
            LlmTurn(content="Reliable knowledge is currently unavailable."),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(
                run_request().model_copy(update={"knowledge_capability_token": None})))
        self.assertEqual(result.tools_used[0].status, "error")


class SkillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "skills"
        cls.tools = ToolRegistry.native_tool_names() | {"search_medical_knowledge"}

    def test_catalog_and_examples(self):
        catalog = SkillCatalog(self.root, self.tools)
        self.assertEqual(catalog.count, 3)
        for directory in self.root.iterdir():
            if directory.is_dir():
                import json
                examples = json.loads((directory / "examples.json").read_text())
                self.assertGreaterEqual(len(examples), 8)
        for summary in catalog.summaries("doctor"):
            doctor = catalog.activate(summary["id"], "doctor")
            patient = catalog.activate(summary["id"], "patient")
            self.assertNotEqual(doctor.instructions, patient.instructions)

    def test_catalog_prompt_explicitly_requires_activation_before_skill_workflow(self):
        catalog = SkillCatalog(self.root, self.tools)
        registry = ToolRegistry(FakeBackend(), FakeKnowledge(), catalog)
        prompt = registry.skill_catalog_prompt("doctor")
        self.assertIn("call activate_skill before domain tools", prompt)
        self.assertIn("explicitly asks to use a workflow or skill", prompt)
        knowledge = next(item for item in registry.definitions()
                         if item["function"]["name"] ==
                         "search_medical_knowledge")
        self.assertIn("activate the matching skill first",
                      knowledge["function"]["description"])

    def test_full_explanation_workflow_requires_boundary_evidence(self):
        catalog = SkillCatalog(self.root, self.tools)
        for role in ("patient", "doctor"):
            activation = catalog.activate("explain_prediction", role)
            self.assertIn("full workflow", activation.instructions)
            self.assertIn("search_medical_knowledge", activation.instructions)

    def test_all_24_packaged_scenarios_follow_declared_tools(self):
        import json
        catalog = SkillCatalog(self.root, self.tools)
        first = "pred_" + "a" * 32
        second = "pred_" + "b" * 32
        citation = "cite_" + "a" * 20
        arguments = {
            "get_prediction": {"prediction_id": first},
            "get_explanation": {"prediction_id": first},
            "get_prediction_history": {"limit": 5},
            "compare_predictions": {
                "prediction_id_a": first, "prediction_id_b": second},
            "search_medical_knowledge": {
                "query": "PPH evidence", "scope": "all", "top_k": 5},
        }
        staged_tools = {
            "compare_prediction_history": [
                "get_prediction_history", "compare_predictions"],
            "explain_prediction": [
                "get_prediction", "get_explanation",
                "search_medical_knowledge"],
            "pph_evidence_education": ["search_medical_knowledge"],
        }
        executed = 0
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            skill_id = directory.name
            examples = json.loads((directory / "examples.json").read_text())
            for number, example in enumerate(examples):
                tools = staged_tools[skill_id]
                calls = [LlmTurn(tool_calls=[LlmToolCall(
                    id=f"t{index}", name=name, arguments=arguments[name])])
                    for index, name in enumerate(tools)]
                prediction_ids = []
                if any(name not in {
                        "search_medical_knowledge",
                        "get_prediction_history"} for name in tools):
                    prediction_ids = [first]
                    if "compare_predictions" in tools:
                        prediction_ids.append(second)
                source_ids = ([citation] if
                              "search_medical_knowledge" in tools else [])
                mentions = " ".join(prediction_ids + source_ids) or "general answer"
                llm = ScriptedLlmClient([
                    LlmTurn(tool_calls=[LlmToolCall(
                        id="activate", name="activate_skill",
                        arguments={"skill_id": skill_id})]),
                    *calls,
                    LlmTurn(content=mentions,
                            grounding_prediction_ids=prediction_ids,
                            grounding_source_ids=source_ids),
                ])
                scenario = run_request().model_copy(update={
                    "message": f"使用 {skill_id} Skill：{example['input']}",
                    "run_id": "run_" + format(number + executed + 1, "032x")[-32:]})
                result = asyncio.run(AgentLoop(
                    llm, ToolRegistry(
                        FakeBackend(history_count=2), FakeKnowledge(),
                        catalog)).run(
                        scenario))
                self.assertEqual(
                    [item.name for item in result.tools_used[1:]],
                    tools)
                self.assertEqual(result.skill_used.id, skill_id)
                executed += 1
        self.assertGreaterEqual(executed, 24)

    def test_progressive_activation_and_persistence_metadata(self):
        catalog = SkillCatalog(self.root, self.tools)
        prediction = "pred_" + "a" * 32
        citation = "cite_" + "a" * 20
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="s1", name="activate_skill",
                arguments={"skill_id": "explain_prediction"})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="p1", name="get_prediction",
                arguments={"prediction_id": prediction})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="e1", name="get_explanation",
                arguments={"prediction_id": prediction})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={
                    "query": "treeSem explanation", "scope": "model",
                    "top_k": 5})]),
            LlmTurn(content=f"Explanation for {prediction}. {citation}",
                    grounding_prediction_ids=[prediction],
                    grounding_source_ids=[citation]),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge(), catalog)).run(
                run_request().model_copy(update={
                    "message": "使用预测解释技能"})))
        self.assertEqual(result.skill_used.id, "explain_prediction")
        self.assertEqual(result.skill_used.catalog_version, catalog.version)
        self.assertTrue(any("<skill>" in (item.get("content") or "")
                            for item in llm.requests[1]))

    def test_active_skill_cannot_call_undeclared_tool(self):
        catalog = SkillCatalog(self.root, self.tools)
        backend = FakeBackend()
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="s1", name="activate_skill",
                arguments={"skill_id": "pph_evidence_education"})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="x1", name="predict_sample", arguments={"sample_index": 0})]),
        ])
        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(AgentLoop(
                llm, ToolRegistry(backend, FakeKnowledge(), catalog)).run(
                    run_request().model_copy(update={
                        "message": "使用PPH循证教育技能"})))
        self.assertEqual(caught.exception.code, "tool_not_allowed")
        self.assertEqual(backend.calls, [])

    def test_patient_cannot_activate_doctor_only_skill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "doctor_notes"
            package.mkdir()
            (package / "skill.yaml").write_text(
                '{"schema_version":1,"id":"doctor_notes","version":"1.0.0",'
                '"display_name":"Doctor notes","description":"Doctor only",'
                '"audiences":["doctor"],"intent_examples":["review"],'
                '"required_tools":["get_prediction"],'
                '"required_knowledge_scopes":["model_technical"],'
                '"instruction_files":{"doctor":"instructions.doctor.md"}}')
            (package / "instructions.doctor.md").write_text("Use verified facts.")
            (package / "examples.json").write_text("[]")
            catalog = SkillCatalog(root, self.tools)
            with self.assertRaises(ValueError):
                catalog.activate("doctor_notes", "patient")

    def test_patient_skill_cannot_require_professional_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "unsafe_scope"
            package.mkdir()
            (package / "skill.yaml").write_text(
                '{"schema_version":1,"id":"unsafe_scope","version":"1.0.0",'
                '"display_name":"Unsafe","description":"Invalid scope",'
                '"audiences":["patient"],"intent_examples":["review"],'
                '"required_tools":["search_medical_knowledge"],'
                '"required_knowledge_scopes":["clinical_professional"],'
                '"instruction_files":{"patient":"instructions.patient.md"}}')
            (package / "instructions.patient.md").write_text("Use verified facts.")
            (package / "examples.json").write_text("[]")
            with self.assertRaises(ValueError):
                SkillCatalog(root, self.tools)

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad").symlink_to(self.root / "explain_prediction",
                                      target_is_directory=True)
            with self.assertRaises(ValueError):
                SkillCatalog(root, self.tools)


if __name__ == "__main__":
    unittest.main()
