"""复核旁表必须绑定正文，程序证据不能放行标注。"""

import copy
import hashlib
import json

import pytest


def corpus():
    return {"metadata": {"dataset_version": "fixture"}, "chapters": [{"chapter": "第一章", "articles": [
        {"article_number": "第一条", "content": "正文甲。", "annotations": {k: {"value": [], "review_status": "pending"} for k in ("title", "charges", "elements", "penalty")}},
        {"article_number": "第二条", "content": "正文乙。", "annotations": {k: {"value": [], "review_status": "pending"} for k in ("title", "charges", "elements", "penalty")}},
    ]}]}


def note():
    return {"original_annotations_sha256": "sha256:" + hashlib.sha256(json.dumps(corpus()["chapters"][0]["articles"][0]["annotations"], ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "article_number": "第一条", "content_sha256": "sha256:" + hashlib.sha256("正文甲。".encode()).hexdigest(),
            "findings": ["逐项核对后建议修正"], "sources": [{"url": "https://www.court.gov.cn/example", "locator": "第一条"}],
            "proposal": {"title": "一般规定", "charges": [], "elements": {"kind": "not_a_charge", "branches": []}, "penalty": {"kind": "not_applicable", "rules": []}},
            "unresolved": ["待确认"]}


def test_pdf_boundaries_reject_substring_false_positive_and_keep_pages():
    from evaluation.review_annotations import compare_article_boundaries
    x = corpus()
    result = compare_article_boundaries(x, ["第一条 正文甲。多出的句子。\n第二条 正文乙。"])
    assert result["exact_matched"] == 1
    assert result["records"][0]["status"] == "mismatch"
    assert result["records"][1]["pdf_pages"] == [1]
    assert result["legal_review_completed"] is False


def test_pdf_article_split_across_pages_and_section_heading():
    from evaluation.review_annotations import compare_article_boundaries
    result = compare_article_boundaries(corpus(), ["－1－\n第一条 正文", "－2－\n甲。\n第二章 第二部分\n第二条 正文乙。"])
    assert result["exact_matched"] == 2
    assert result["records"][0]["pdf_pages"] == [1, 2]


def test_duplicate_or_missing_anchor_is_not_verified():
    from evaluation.review_annotations import compare_article_boundaries
    result = compare_article_boundaries(corpus(), ["第一条 正文甲。\n第一条 正文甲。"])
    assert result["exact_matched"] == 0
    assert result["records"][0]["status"] == "duplicate_anchor"
    assert result["records"][1]["status"] == "missing_anchor"


def test_review_sidecar_does_not_mutate_or_promote_runtime():
    from evaluation.review_annotations import build_review
    x = corpus()
    before = copy.deepcopy(x)
    result = build_review(x, [note()], [], "sha256:input")
    assert x == before
    assert result["summary"]["agent_reviewed"] == 1
    assert result["summary"]["pending_agent_review"] == 1
    assert result["summary"]["human_accepted"] == 0
    assert result["summary"]["new_coverage_eligible"] == 0
    assert result["records"][0]["human_acceptance"] == "pending"
    assert result["records"][1]["agent_review_status"] == "pending"


@pytest.mark.parametrize("change", ["hash", "incomplete", "unknown", "duplicate", "authority"])
def test_notes_with_wrong_binding_or_missing_review_are_rejected(change):
    from evaluation.review_annotations import build_review
    notes = [note()]
    if change == "hash":
        notes[0]["content_sha256"] = "sha256:stale"
    elif change == "incomplete":
        del notes[0]["proposal"]["elements"]
    elif change == "unknown":
        notes[0]["article_number"] = "第三条"
    elif change == "duplicate":
        notes.append(copy.deepcopy(notes[0]))
    else:
        notes[0]["sources"][0]["url"] = "https://www.court.gov.cn.attacker.example/article"
    with pytest.raises(ValueError):
        build_review(corpus(), notes, [], "sha256:input")


def test_appendix_and_final_provisions_are_not_article_body():
    from evaluation.review_annotations import compare_article_boundaries
    result = compare_article_boundaries(corpus(), ["第一条 正文甲。\n附 则\n第二条 正文乙。\n附件一\n1．外部文件"])
    assert result["exact_matched"] == 2


def test_unchanged_body_with_changed_original_annotations_rejects_stale_review():
    from evaluation.review_annotations import build_review
    x = corpus()
    n = note()
    n["original_annotations_sha256"] = "sha256:" + hashlib.sha256(json.dumps(x["chapters"][0]["articles"][0]["annotations"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    x["chapters"][0]["articles"][0]["annotations"]["title"]["value"] = "changed"
    with pytest.raises(ValueError):
        build_review(x, [n], [], "sha256:input")


def test_delivered_review_keeps_all_pending_annotations_outside_runtime():
    from pathlib import Path

    from app.knowledge.law_knowledge import _build_article_index, is_lawref_eligible

    from evaluation.review_annotations import build_review, fingerprint

    root = Path(__file__).resolve().parents[1]
    raw = (root / "backend/data/law_knowledge/criminal_law_full.json").read_bytes()
    x = json.loads(raw)
    notes = json.loads((root / "docs/knowledge/law_review/2026-09-30.batch1.notes.json").read_text())
    delivered = json.loads((root / "docs/knowledge/law_review/2026-09-30.review.json").read_text())
    assert notes["corpus_sha256"] == delivered["corpus_sha256"] == fingerprint(raw)
    assert build_review(x, notes["records"], [r["text_check"] for r in delivered["records"]], fingerprint(raw))["records"] == delivered["records"]
    assert sum(is_lawref_eligible(a) for a in _build_article_index(x).values()) == 6
    assert all(r["human_acceptance"] == "pending" and not r["coverage_eligible"] for r in delivered["records"])
    reviewed = {r["article_number"]: r for r in delivered["records"] if r["agent_review_status"] != "pending"}
    assert "强令、组织他人违章冒险作业罪" in reviewed["第一百三十四条"]["proposal"]["charges"]
    assert reviewed["第四百零八条之一"]["proposal"]["title"] == "食品、药品监管渎职罪"
    danger = reviewed["第一百三十三条之一"]["proposal"]["elements"]
    assert "情节恶劣" not in "".join(danger["common_conditions"])
    assert len(danger["alternative_branches"]) == 5
    assert reviewed["第一百四十九条"]["proposal"]["charges"] == []
    assert reviewed["第一百九十九条"]["proposal"]["elements"]["kind"] == "repealed"
    assert delivered["text_evidence"]["exact_matched"] == 505
