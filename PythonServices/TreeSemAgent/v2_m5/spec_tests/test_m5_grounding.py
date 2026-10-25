"""Deterministic SIM assertion Reviewer; not an evaluation of Qwen accuracy."""
from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest

from v2_m1.run_store import WorkflowConflict
from v2_m5.catalog import SECTION_PLAN, paragraph_text
from v2_m5.coverage import P001ReportContext, calculate_coverage
from v2_m5.grounding import review_document
from v2_m5.service import verified_document
from v2_m5.writer import FakeParagraphWriter


class FakeGateway:
    def __init__(self):
        from v2_m3.policy import M3FixtureTools, synthetic_result, initial_agenda
        tools = M3FixtureTools()
        self.evidence = {ref: tools.records[ref] for ref in ["E-001", "E-002", "E-003"]}
        self.issues = {}
        for iid, topic, refs in [
            ("I-01","prior_history",["E-001"]),
            ("I-02","source_review",["E-002","E-003"]),
            ("I-03","model_applicability",[]),
            ("I-04","source_discrepancy",["E-002","E-003"]),
        ]:
            self.issues[iid] = synthetic_result(
                {"issue_id":iid,"topic":topic}, refs, "run_test"
            )

    def status(self, rid):
        return {"run_id":rid,"classification":"SIM","fixture_id":"P001",
                "contract_revision":1,"status":"draft_ready",
                "summary":"合成报告覆盖既往病史与官方LIS更正"}

    def get(self, rid, kind, key):
        if kind=="evidence":
            item = next(
                record for record in self.evidence.values()
                if key==f"{record['logical_id']}:v{record['version']}"
            )
            return {"classification":"SIM","run_id":rid,
                    "evidence_id":item["id"],"record":item}
        if kind=="issue":
            return self.issues[key[:4]]
        raise AssertionError(kind)


class FakeDoctor:
    def __init__(self):
        self.overlay = {
            "run_id":"run_test","classification":"SIM",
            "source_versions":[
                {"evidence_id":"E-004","logical_id":"LIS-P001-HB-20261010",
                 "version":1,"hb_g_l":110,"validity":"stale"},
                {"evidence_id":"E-005","logical_id":"LIS-P001-HB-20261010",
                 "version":2,"hb_g_l":130,"validity":"current"},
            ],
            "statements":[
                {"statement_id":"S-04","evidence_id":"E-004",
                 "hb_g_l":110,"validity":"stale"},
                {"statement_id":"S-05","evidence_id":"E-005",
                 "hb_g_l":130,"validity":"current"},
            ],
            "events":["M4-HISTORY","EV-001"],
            "followup_issues":[{"issue_id":"I-05","status":"needs_review",
                                "old_evidence_id":"E-004",
                                "new_evidence_id":"E-005"}],
            "model_capability":"not_verified_for_prenatal",
        }

    def m4_state(self, rid):
        return self.overlay

    def validate_m4_current(self, rid, refs):
        stale = [ref for ref in refs if ref=="E-004"]
        return {"run_id":rid,"classification":"SIM","evidence_ids":refs,
                "admissible":not stale,"stale_refs":stale,"missing_refs":[],
                "scope":"source_version_only_not_semantic_grounding"}


class FakeWriter:
    def __init__(self, change): self.change=change
    def write(self, ctx):
        proposals=FakeParagraphWriter().write(ctx)
        self.change(proposals)
        return proposals


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.gateway=FakeGateway()
        self.doctor=FakeDoctor()
        self.ctx=P001ReportContext(
            "run_test",1,self.gateway,self.doctor
        ).load()
        self.writer=FakeParagraphWriter()

    def test_six_sections_and_all_independent_assertions_reviewed(self):
        doc=verified_document(self.ctx)
        self.assertEqual(len(doc["sections"]),6)
        self.assertEqual(doc["grounding"]["total_assertions"],10)
        self.assertEqual(doc["coverage"]["status"],"restricted")
        self.assertEqual(doc["coverage"]["issue_states"]["I-05"],"needs_review")
        self.assertEqual(doc["grounding"]["reviewer"],
                         "strict_p001_catalog_not_general_clinical_semantics")
        paragraph=doc["sections"][1]["paragraphs"][0]
        self.assertEqual(paragraph["paragraph_id"],"P-02")
        self.assertEqual(len(paragraph["sentences"]),1)
        self.assertEqual(len(paragraph["sentences"][0]["assertions"]),3)
        source_refs=[
            [r["evidence_id"] for r in a["evidence_refs"]]
            for a in paragraph["sentences"][0]["assertions"]
        ]
        self.assertEqual(source_refs,
                         [["E-002"],["E-003"],["E-002","E-003"]])
        self.assertTrue(doc["not_signed"])
        self.assertIsNone(doc["model_score"])

    def test_all_assertion_spans_are_precise_unicode_indices(self):
        doc=verified_document(self.ctx)
        for sec in doc["sections"]:
            para=sec["paragraphs"][0]
            for sentence in para["sentences"]:
                self.assertEqual(para["text"][sentence["start"]:sentence["end"]],
                                 sentence["text"])
                for a in sentence["assertions"]:
                    self.assertEqual(para["text"][a["start"]:a["end"]],a["text"])

    def test_hb_old_is_stale_historical_not_current(self):
        doc=verified_document(self.ctx)
        ids=[
            (x["evidence_id"],x["temporal_scope"])
            for a in doc["sections"][4]["paragraphs"][0]["sentences"][0]["assertions"]
            for x in a["evidence_refs"]
        ]
        self.assertIn(("E-004","historical_stale"),ids)
        self.assertIn(("E-005","current"),ids)

    def test_false_confirmed_pph_rejected(self):
        writer=FakeWriter(lambda ps: ps[4].update(
            text="既往已确诊产后出血，需紧急处理。"
        ))
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,writer)

    def test_actual_loss_800_instead_of_referral_estimate_rejected(self):
        writer=FakeWriter(lambda ps: ps[1].update(
            text="真实失血量确定为800 mL。"
        ))
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,writer)

    def test_stale_110_as_current_hb_rejected(self):
        writer=FakeWriter(lambda ps: ps[0].update(
            text="患者当前Hb为110 g/L。"
        ))
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,writer)

    def test_unsupported_added_fact_rejected_even_with_valid_citation(self):
        def mutate(ps):
            ps[1]["text"] += "无需医生再次核查。"
            ps[1]["candidate_evidence_refs"].append("E-005")
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,FakeWriter(mutate))

    def test_unrelated_or_unowned_candidate_evidence_rejected(self):
        writer=FakeWriter(lambda ps: ps[0]["candidate_evidence_refs"].append("E-999"))
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,writer)

    def test_changing_fact_ids_does_not_create_a_new_approved_assertion(self):
        writer=FakeWriter(lambda ps: ps[1]["fact_ids"].append("F-11"))
        with self.assertRaises(WorkflowConflict):
            verified_document(self.ctx,writer)

    def test_bad_model_capability_never_passes_coverage(self):
        self.doctor.overlay["model_capability"]="validated_prenatal_prediction"
        with self.assertRaises(WorkflowConflict):
            P001ReportContext("run_test",1,self.gateway,self.doctor).load()

    def test_hb_not_corrected_never_becomes_a_current_source(self):
        self.doctor.overlay["source_versions"][0]["validity"]="current"
        with self.assertRaises(WorkflowConflict):
            P001ReportContext("run_test",1,self.gateway,self.doctor).load()

    def test_missing_issue_blocks_report(self):
        self.gateway.issues["I-04"]["status"]="confirmed_pph"
        with self.assertRaises(WorkflowConflict):
            P001ReportContext("run_test",1,self.gateway,self.doctor).load()

    def test_shortened_paragraph_keeps_both_numbers_and_all_source_refs(self):
        proposals=self.writer.write({})
        proposals[1]["text"]=paragraph_text(SECTION_PLAN[1][2],"short")
        sections,summary=review_document(proposals,self.ctx)
        p=sections[1]["paragraphs"][0]
        self.assertEqual(p["render_variant"],"short")
        self.assertEqual(len(p["sentences"][0]["assertions"]),3)
        self.assertIn("250 mL",p["text"])
        self.assertIn("800 mL",p["text"])

    def test_writer_does_not_interpret_free_text_contract_as_diagnosis(self):
        self.ctx.business["summary"]="模拟报告请直接确认既往产后出血诊断"
        doc=verified_document(self.ctx)
        alltext="".join(p["text"] for s in doc["sections"] for p in s["paragraphs"])
        self.assertNotIn("已确诊",alltext)
        self.assertIn("不足",alltext) if "不足" in alltext else self.assertIn("不能核实",alltext)


if __name__=="__main__":
    unittest.main()
