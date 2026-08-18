import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.llm_client import ScriptedLlmClient
from agent.loop import AgentExecutionError, AgentLoop
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

    def test_fabricated_citation_is_rejected(self):
        llm = ScriptedLlmClient([LlmTurn(
            content="Unsupported [cite_" + "f" * 20 + "].",
            grounding_source_ids=["cite_" + "f" * 20])])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(
                llm, ToolRegistry(FakeBackend(), FakeKnowledge())).run(run_request()))

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
        executed = 0
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            skill_id = directory.name
            examples = json.loads((directory / "examples.json").read_text())
            for number, example in enumerate(examples):
                calls = [LlmToolCall(
                    id=f"t{index}", name=name, arguments=arguments[name])
                    for index, name in enumerate(example["expected_tools"])]
                prediction_ids = []
                if any(name != "search_medical_knowledge"
                       for name in example["expected_tools"]):
                    prediction_ids = [first]
                    if "compare_predictions" in example["expected_tools"]:
                        prediction_ids.append(second)
                source_ids = ([citation] if "search_medical_knowledge" in
                              example["expected_tools"] else [])
                mentions = " ".join(prediction_ids + source_ids) or "general answer"
                llm = ScriptedLlmClient([
                    LlmTurn(tool_calls=[LlmToolCall(
                        id="activate", name="activate_skill",
                        arguments={"skill_id": skill_id})]),
                    LlmTurn(tool_calls=calls),
                    LlmTurn(content=mentions,
                            grounding_prediction_ids=prediction_ids,
                            grounding_source_ids=source_ids),
                ])
                scenario = run_request().model_copy(update={
                    "message": example["input"],
                    "run_id": "run_" + format(number + executed + 1, "032x")[-32:]})
                result = asyncio.run(AgentLoop(
                    llm, ToolRegistry(FakeBackend(), FakeKnowledge(), catalog)).run(
                        scenario))
                self.assertEqual(
                    [item.name for item in result.tools_used[1:]],
                    example["expected_tools"])
                self.assertEqual(result.skill_used.id, skill_id)
                executed += 1
        self.assertGreaterEqual(executed, 24)

    def test_progressive_activation_and_persistence_metadata(self):
        catalog = SkillCatalog(self.root, self.tools)
        prediction = "pred_" + "a" * 32
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="s1", name="activate_skill",
                arguments={"skill_id": "explain_prediction"})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="p1", name="get_explanation",
                arguments={"prediction_id": prediction})]),
            LlmTurn(content=f"Explanation for {prediction}.",
                    grounding_prediction_ids=[prediction]),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge(), catalog)).run(
                run_request()))
        self.assertEqual(result.skill_used.id, "explain_prediction")
        self.assertEqual(result.skill_used.catalog_version, catalog.version)
        self.assertTrue(any("<skill>" in (item.get("content") or "")
                            for item in llm.requests[1]))

    def test_active_skill_cannot_call_undeclared_tool(self):
        catalog = SkillCatalog(self.root, self.tools)
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="s1", name="activate_skill",
                arguments={"skill_id": "pph_evidence_education"})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="x1", name="predict_sample", arguments={"sample_index": 0})]),
            LlmTurn(content="The undeclared operation was rejected."),
        ])
        result = asyncio.run(AgentLoop(
            llm, ToolRegistry(FakeBackend(), FakeKnowledge(), catalog)).run(
                run_request()))
        self.assertEqual(result.tools_used[-1].status, "error")

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
