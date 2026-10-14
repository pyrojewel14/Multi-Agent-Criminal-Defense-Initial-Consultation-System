"""RRF preserves raw channel evidence without qualification laundering."""

import pytest

from app.knowledge.retrieval.fusion import document_key, rrf_fuse
from app.knowledge.retrieval.models import CandidateOrigin, ChannelResult, RetrievalCandidate, StageStatus


def candidate(cid, row_id=None, source="one", **metadata):
    return RetrievalCandidate(cid, source, f"body-{cid}", {"row_id": row_id or cid, **metadata})


def channel(name, candidates, scores=None):
    return ChannelResult(name, tuple(candidates), scores or {}, StageStatus("executed"))


def choose(items):
    return min(items, key=lambda c: c.candidate_id)


def test_rrf_uses_rank_and_preserves_each_raw_score():
    target = candidate("target")
    result = rrf_fuse({
        "vector": channel("vector", [target], {"target": 0.8}),
        "bm25": channel("bm25", [candidate("other"), target], {"target": 7.5}),
    }, key=document_key, choose_representative=choose)
    assert result[0].candidate_id == "target"
    assert result[0].fusion_score == pytest.approx(1 / 61 + 1 / 62)
    assert result[0].recall_ranks == {"vector": 1, "bm25": 2}
    assert result[0].recall_scores == {"vector": 0.8, "bm25": 7.5}
    assert {o.channel for o in result[0].origins} == {"vector", "bm25"}


def test_same_channel_duplicates_only_contribute_once_and_ties_are_stable():
    a, b = candidate("a"), candidate("b")
    result = rrf_fuse({"x": channel("x", [b, b]), "y": channel("y", [a])}, key=document_key, choose_representative=choose)
    assert [c.candidate_id for c in result] == ["a", "b"]
    assert result[1].fusion_score == pytest.approx(1 / 61)


def test_default_document_identity_keeps_sources_and_rows_distinct():
    items = [candidate("a", "same", "one"), candidate("b", "same", "two"), candidate("c", "different", "one")]
    result = rrf_fuse({"x": channel("x", items)}, key=document_key, choose_representative=choose)
    assert len(result) == 3
    assert document_key(candidate("x", "b:c", "a")) != document_key(candidate("y", "c", "a:b"))


def test_representative_choice_preserves_failed_original_origin():
    failed = RetrievalCandidate("bad", "law", "mismatched", {"article": "264", "verified": False}, origins=(CandidateOrigin("law", "bad", "vector", {"verified": False}),))
    trusted = RetrievalCandidate("good", "law", "official", {"article": "264", "verified": True}, origins=(CandidateOrigin("law", "good", "keyword", {"verified": True}),))
    result = rrf_fuse({"vector": channel("vector", [failed]), "keyword": channel("keyword", [trusted])}, key=lambda c: c.source_id + c.metadata["article"], choose_representative=lambda items: next(c for c in items if c.metadata["verified"]))
    assert len(result) == 1
    assert result[0].content == "official"
    assert result[0].metadata["verified"] is True
    assert [(o.candidate_id, o.metadata["verified"]) for o in result[0].origins] == [("bad", False), ("good", True)]
    assert failed.recall_ranks == {} and failed.metadata["verified"] is False


def test_separate_channels_same_origin_keep_original_channel():
    item = RetrievalCandidate("id", "one", "body", {}, origins=(CandidateOrigin("one", "id", "vector", {"row_id": "row"}),))
    result = rrf_fuse({"hyde": channel("hyde", [item])}, key=document_key, choose_representative=choose)
    assert result[0].origins[0].channel == "vector"
    assert result[0].recall_ranks == {"hyde": 1}
