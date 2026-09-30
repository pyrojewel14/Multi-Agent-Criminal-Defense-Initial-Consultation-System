"""PDF verification is text evidence, never legal-review promotion."""


def test_text_check_removes_page_marks_and_reports_mismatch():
    from evaluation.verify_full_text import compare_text
    corpus = {"chapters": [{"articles": [{"article_number": "第一条", "content": "法律明文规定"}, {"article_number": "第二条", "content": "不在PDF"}]}]}
    result = compare_text(corpus, "第一条 法律\n－2－\n明文 规定。")
    assert result["content_matched"] == 1
    assert result["unmatched"] == ["第二条"]
    assert result["legal_review_completed"] is False
