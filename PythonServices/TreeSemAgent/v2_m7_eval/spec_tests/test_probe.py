"""M7-B CI uses scripted synthetic outputs, NOT real LLM inference."""
import json
from pathlib import Path
import tempfile
import unittest

from v2_m5.writer import FakeParagraphWriter
from v2_m7_eval.run_probe import (
    Completion, FixtureChat, LocalChat, cases, evaluate_agent,
    evaluate_writer, fixture_client, local_endpoint, probe,
    write_report, writer_prompt,
)

def answer(value):
    return Completion(value,"fixture-not-real",0.0,{
        "prompt_tokens":None,"completion_tokens":None,"total_tokens":None})

class ProbeContracts(unittest.TestCase):
    def test_fixture_six_decisions_and_writer(self):
        report=probe(fixture_client(),"fixture")
        self.assertEqual(report["cases"],7)
        self.assertEqual(report["golden_matches"],6)
        self.assertTrue(report["writer_catalog_pass"])
        self.assertFalse(report["real_llm_called"])
        self.assertTrue(all(row["outcome"] in ("golden_match","catalog_pass")
                            for row in report["records"]))
        self.assertIsNone(report["records"][0]["usage"]["prompt_tokens"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"report.json"
            write_report(path,report)
            self.assertEqual(json.loads(path.read_text())["mode"],"fixture")

    def test_agent_parse_and_guard_diagnostics(self):
        case=cases()[0]
        self.assertEqual(evaluate_agent(case,answer("not json"))["outcome"],
                         "invalid_json")
        self.assertEqual(evaluate_agent(case,answer("[]"))["outcome"],
                         "schema_mismatch")
        rejected=evaluate_agent(case,answer(
            '{"action":"tool","tool":"delete_patient"}'))
        self.assertEqual(rejected["outcome"],"guard_rejected")
        self.assertFalse(rejected["guard_pass"])
        accepted=evaluate_agent(case,answer(json.dumps(case["golden"])))
        self.assertEqual(accepted["outcome"],"golden_match")

    def test_writer_reject_not_equivalent_to_semantic_wrong(self):
        proposals=FakeParagraphWriter().write(writer_prompt())
        proposals[1]["text"]="原始记录估计250 mL；转诊摘要记载约800 mL；差异尚未核实。"
        outcome=evaluate_writer(answer(json.dumps(proposals,ensure_ascii=False)))
        self.assertEqual(outcome["outcome"],"catalog_rejected")
        self.assertTrue(outcome["schema_pass"])
        self.assertEqual(outcome["semantic_correctness"],"not_evaluated")
        proposals=FakeParagraphWriter().write(writer_prompt())
        proposals[4]["text"]="已经确诊产后出血。"
        bad=evaluate_writer(answer(json.dumps(proposals,ensure_ascii=False)))
        self.assertEqual(bad["outcome"],"catalog_rejected")
        valid=evaluate_writer(answer(json.dumps(
            FakeParagraphWriter().write(writer_prompt()),ensure_ascii=False)))
        self.assertEqual(valid["outcome"],"catalog_pass")
        self.assertEqual(valid["semantic_correctness"],"not_evaluated")

    def test_external_hosts_and_ambiguous_urls_rejected(self):
        invalid=(
            "https://example.com/v1/chat/completions",
            "http://127.0.0.1:8000/other",
            "http://localhost/v1/chat/completions",
            "http://user:pass@127.0.0.1:8000/v1/chat/completions",
            "http://127.0.0.1:8000/v1/chat/completions?auth=secret",
        )
        for item in invalid:
            with self.subTest(url=item),self.assertRaises(ValueError):
                local_endpoint(item)
        valid="http://127.0.0.1:8000/v1/chat/completions"
        self.assertEqual(local_endpoint(valid),valid)
        with self.assertRaises(ValueError):
            LocalChat(invalid[0],"Qwen")

    def test_provider_failure_not_reported_as_model_success(self):
        result=probe(FixtureChat([]),"fixture")
        self.assertEqual(result["golden_matches"],0)
        self.assertFalse(result["writer_catalog_pass"])
        self.assertTrue(all(row["outcome"]=="provider_error"
                            for row in result["records"]))

if __name__=="__main__":
    unittest.main()
