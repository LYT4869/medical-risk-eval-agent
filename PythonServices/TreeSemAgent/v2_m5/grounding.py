"""Strict P001 SIM sentence/assertion reviewer.

This is **not** a clinically validated general LLM reviewer. Only statements
from the explicitly synthetic vetted catalog can pass. Writer suggestions
(including source citations) are low trust until independently checked.
"""
from __future__ import annotations

import re

from v2_m1.run_store import WorkflowConflict
from .catalog import FACTS, SECTION_PLAN, candidate_refs, paragraph_text


def prove(fact_id: str, ctx) -> list[dict]:
    claim = FACTS[fact_id]
    ids = claim["evidence_ids"]
    for ref in ids:
        if ref in ("E-001", "E-002", "E-003") and ref not in ctx.evidence:
            raise WorkflowConflict("unproven source ref")
        if ref in ("E-004", "E-005") and ref not in ctx.sources:
            raise WorkflowConflict("missing versioned LIS source")
    old, new = ctx.sources["E-004"], ctx.sources["E-005"]
    original, referral = ctx.evidence["E-002"], ctx.evidence["E-003"]
    if (
        ctx.evidence["E-001"].get("kind") != "patient_report" or
        original.get("kind") != "original_delivery" or
        referral.get("kind") != "referral_summary" or
        original.get("blood_loss_ml") != 250 or
        referral.get("blood_loss_ml") != 800 or
        referral.get("approximate") is not True or
        original.get("logical_id") == referral.get("logical_id") or
        ctx.issues["I-04"]["status"] != "closed_unresolved"
    ):
        raise WorkflowConflict("source conflict provenance invalid")
    if fact_id in ("F-05",) and (
        ctx.issues["I-03"]["status"] != "blocked" or
        ctx.overlay["model_capability"] != "not_verified_for_prenatal"
    ):
        raise WorkflowConflict("model not blocked")
    if fact_id in ("F-06", "F-07") and (
        old["hb_g_l"] != 110 or new["hb_g_l"] != 130 or
        old["logical_id"] != new["logical_id"] or
        (old["version"], new["version"]) != (1, 2) or
        (old["validity"], new["validity"]) != ("stale", "current") or
        ctx.statements["S-04"]["validity"] != "stale" or
        ctx.statements["S-05"]["validity"] != "current"
    ):
        raise WorkflowConflict("current Hb assertion is unsupported")
    if fact_id in ("F-08",) and (
        ctx.issues["I-01"]["no_diagnosis"] is not True
    ):
        raise WorkflowConflict("unverified diagnosis cannot be confirmed")
    if fact_id not in FACTS:
        raise WorkflowConflict("unsupported assertion ID")
    return [
        {"evidence_id": ref, "source_version": (
            ctx.sources[ref]["version"] if ref in ctx.sources
            else ctx.evidence[ref]["version"]),
         "logical_id": (
            ctx.sources[ref]["logical_id"] if ref in ctx.sources
            else ctx.evidence[ref]["logical_id"]),
         "temporal_scope": (
            "historical_stale" if ref == "E-004" else
            "current" if ref == "E-005" else "historical_T1"),
         "reference_kind": ("LIS_SIM" if ref.startswith("E-00") and
                             ref in ("E-004", "E-005") else "P001_SIM_EMR")}
        for ref in ids
    ]


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    if not isinstance(text, str) or not text or len(text) > 1600:
        raise WorkflowConflict("invalid paragraph text length")
    matches = list(re.finditer(r"[^。！？]+[。！？]", text))
    if "".join(x.group(0) for x in matches) != text:
        raise WorkflowConflict("unprocessed span in Writer paragraph")
    return [(m.start(), m.end(), m.group(0)) for m in matches]


def review_paragraph(proposal: object, section: str, paragraph_id: str,
                     expected_facts: tuple[str, ...], ctx) -> dict:
    if not isinstance(proposal, dict) or (
        set(proposal) != {"section_id", "paragraph_id", "text",
                          "candidate_evidence_refs", "fact_ids"}
    ):
        raise WorkflowConflict("Writer output violates strict paragraph schema")
    if (proposal["section_id"], proposal["paragraph_id"],
            proposal["fact_ids"]) != (section, paragraph_id, list(expected_facts)):
        raise WorkflowConflict("Writer changed trusted issue/section structure")
    refs = proposal["candidate_evidence_refs"]
    if not isinstance(refs, list) or any(
        not isinstance(v, str) or v not in
        {"E-001", "E-002", "E-003", "E-004", "E-005"} for v in refs
    ) or len(refs) != len(set(refs)):
        raise WorkflowConflict("candidate evidence contains unauthorized IDs")
    text = proposal["text"]
    if text == paragraph_text(expected_facts, "full"):
        variant = "full"
    elif text == paragraph_text(expected_facts, "short"):
        variant = "short"
    else:
        raise WorkflowConflict(
            "Writer contains unsupported/unreviewed assertion, numerical "
            "change or added free-form medical text"
        )
    sentences = _split_sentences(text)
    if len(sentences) != 1:
        raise WorkflowConflict("prototype expected one complete P001 sentence")
    start, end, sentence = sentences[0]
    verified = []
    for index, fact_id in enumerate(expected_facts, 1):
        phrase = FACTS[fact_id][variant]
        offset = sentence.find(phrase)
        if offset < 0 or sentence.count(phrase) != 1:
            raise WorkflowConflict("sentence assertion span missing or ambiguous")
        citations = prove(fact_id, ctx)
        verified.append({
            "assertion_id": f"A-{paragraph_id[2:]}-01-{index:02d}",
            "fact_id": fact_id, "text": phrase, "start": start + offset,
            "end": start + offset + len(phrase),
            "status": "supported_sim_catalog",
            "evidence_refs": citations,
        })
    return {
        "section_id": section, "paragraph_id": paragraph_id, "text": text,
        "render_variant": variant, "fact_ids": list(expected_facts),
        "candidate_evidence_refs": refs,
        "grounding_status": "all_assertions_supported_sim_catalog",
        "sentences": [{
            "sentence_id": f"S-{paragraph_id[2:]}-01",
            "start": start, "end": end, "text": sentence,
            "status": "all_assertions_reviewed_sim_catalog",
            "assertions": verified,
        }],
    }


def review_document(proposals: list[dict], ctx):
    if not isinstance(proposals, list) or len(proposals) != len(SECTION_PLAN):
        raise WorkflowConflict("Writer must produce exactly six P001 paragraphs")
    sections = []
    for i, (proposal, (section, pid, facts)) in enumerate(
        zip(proposals, SECTION_PLAN, strict=True)
    ):
        paragraph = review_paragraph(proposal, section, pid, facts, ctx)
        sections.append({"section_id": section, "paragraphs": [paragraph]})
    reviewed = [
        assertion for section in sections
        for p in section["paragraphs"]
        for sentence in p["sentences"]
        for assertion in sentence["assertions"]
    ]
    if len(reviewed) != sum(len(facts) for _, _, facts in SECTION_PLAN):
        raise WorkflowConflict("not all assertions have been reviewed")
    return sections, {
        "reviewer": "strict_p001_catalog_not_general_clinical_semantics",
        "total_sentences": len(SECTION_PLAN),
        "total_assertions": len(reviewed),
        "supported_assertions": len(reviewed),
        "unsupported_assertions": 0,
        "admission": "restricted_only",
    }
