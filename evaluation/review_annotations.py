"""生成绑定原语料的复核旁表；程序核对和代理建议均不放行运行时标注。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ARTICLE = re.compile(r"^\s*(第[一二三四五六七八九十百千零]+条(?:之[一二三四五六七八九十]+)?)\s+(.+)$")
HEADING = re.compile(r"^第[一二三四五六七八九十]+[编章节]\s")
FIELDS = {"title", "charges", "elements", "penalty"}


def fingerprint(raw: bytes) -> str:
    """生成与正文资产一致的指纹格式。"""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def articles(corpus: dict) -> list[dict]:
    """按原始章节顺序展开条文。"""
    return [a for c in corpus["chapters"] for a in c["articles"]]


def compact(text: str) -> str:
    """仅忽略排版空白，保留所有标点和正文字符。"""
    return re.sub(r"\s+", "", text)


def compare_article_boundaries(corpus: dict, pages: list[str]) -> dict:
    """按独立条号锚点与正文边界比较，防止子串命中掩盖漏句。"""
    extracted: dict[str, list[dict]] = {}
    current = None
    for page_number, page in enumerate(pages, 1):
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"－\s*\d+\s*－", line):
                continue
            match = ARTICLE.match(line)
            if match:
                number, body = match.groups()
                current = {"parts": [body], "pages": [page_number]}
                extracted.setdefault(number, []).append(current)
            elif HEADING.match(line) or re.fullmatch(r"附\s*则|附件[一二]", line):
                # 章节标题终止上一条正文，避免目录或章节名混入正文。
                current = None
            elif current is not None:
                current["parts"].append(line)
                if page_number not in current["pages"]:
                    current["pages"].append(page_number)
    records = []
    for article in articles(corpus):
        number = article["article_number"]
        hits = extracted.get(number, [])
        status = "missing_anchor" if not hits else "duplicate_anchor" if len(hits) != 1 else "mismatch"
        hit = hits[0] if len(hits) == 1 else None
        body = "".join(hit["parts"]) if hit else ""
        if hit and compact(body) == compact(article["content"]):
            status = "exact_match"
        record = {"article_number": number, "status": status, "pdf_pages": hit["pages"] if hit else [],
                  "content_sha256": fingerprint(article["content"].encode()),
                  "normalized_pdf_sha256": fingerprint(compact(body).encode()) if hit else None}
        if status == "mismatch":
            record["extracted_text"] = body
        records.append(record)
    return {"records": records, "exact_matched": sum(r["status"] == "exact_match" for r in records),
            "legal_review_completed": False, "evidence_kind": "supplied_pdf_article_boundary_check"}


def build_review(corpus: dict, notes: list[dict], text_records: list[dict], corpus_sha256: str) -> dict:
    """合并逐条代理意见和程序证据，保留原值与待接受状态。"""
    source_articles = articles(corpus)
    index = {a["article_number"]: a for a in source_articles}
    checked = {}
    for note in notes:
        number = note["article_number"]
        if number not in index or number in checked:
            raise ValueError("复核条号未知或重复: " + number)
        if note["content_sha256"] != fingerprint(index[number]["content"].encode()):
            raise ValueError("复核绑定的正文已变化: " + number)
        original_hash = fingerprint(json.dumps(index[number]["annotations"], ensure_ascii=False, sort_keys=True).encode())
        if note.get("original_annotations_sha256") != original_hash:
            raise ValueError("复核绑定的原标注已变化: " + number)
        if set(note["proposal"]) != FIELDS or not note.get("findings") or not note.get("unresolved") or not note.get("sources"):
            raise ValueError("复核字段或证据不完整: " + number)
        for source in note["sources"]:
            parsed = urlparse(source["url"])
            host = parsed.hostname or ""
            if parsed.scheme != "https" or not host.endswith(".gov.cn") or parsed.username or parsed.password or not source.get("locator"):
                raise ValueError("复核来源不是政府 HTTPS 地址或缺少定位: " + number)
        checked[number] = note
    text_index = {r["article_number"]: r for r in text_records}
    records = []
    for article in source_articles:
        number = article["article_number"]
        note = checked.get(number)
        record = {"article_number": number, "content_sha256": fingerprint(article["content"].encode()),
                  "original": copy.deepcopy(article["annotations"]), "text_check": text_index.get(number),
                  "agent_review_status": "reviewed_with_proposal" if note else "pending",
                  "human_acceptance": "pending", "lawyer_confirmation": "pending", "coverage_eligible": False}
        if note:
            record.update({k: copy.deepcopy(note[k]) for k in ("findings", "sources", "proposal", "unresolved")})
            record["reviewer"] = "Codex"
            record["reviewed_at"] = "2026-09-30"
        records.append(record)
    return {"schema_version": "annotation-review-proposals-v1", "corpus_sha256": corpus_sha256,
            "dataset_version": corpus["metadata"]["dataset_version"], "legal_review_completed": False,
            "summary": {"records": len(records), "agent_reviewed": len(checked), "pending_agent_review": len(records) - len(checked),
                        "human_accepted": 0, "lawyer_confirmed": 0, "new_coverage_eligible": 0},
            "records": records}


def render_report(review: dict) -> str:
    """输出供人逐条接受或驳回的证据清单。"""
    s = review["summary"]
    lines = ["# 刑法标注复核：第一批（2026-09-30）", "",
             f"已完成 {s['agent_reviewed']} 条代理逐项核对与修正建议；另 {s['pending_agent_review']} 条仍待代理复核。505条全部保留在机器可读旁表中。",
             "这是一批可审阅提议，尚无用户接受或律师确认，不新增运行时覆盖资格。原始全文、六条快照、索引与默认服务配置均未改写。",
             "", "## 复核依据与接受方式", "",
             "正文依据整合至刑法修正案（十二）的政府PDF。程序逐条比较边界与页码，只证明所提供PDF与语料的文本一致性；未认证远端PDF当前字节。罪名依据另列的两高文件。",
             "分支模型保留共同条件、替代条件、特殊主体、引用、竞合和刑罚档次；不是已经认证的完整构成要件表。主观要素、责任能力、司法解释阈值和行为时法仍须补核。",
             "审阅者可按条号回复“接受建议／退回及理由／律师确认待补”。接受报告不直接写回运行时资产，放行前须另行检验分支模型与覆盖计算的契约。",
             "", "## 本批清单", "", "| 条号 | 主要问题 |", "| --- | --- |"]
    reviewed = [r for r in review["records"] if r["agent_review_status"] != "pending"]
    for r in reviewed:
        lines.append(f"| {r['article_number']} | {'；'.join(r['findings'])} |")
    for r in reviewed:
        p = r["proposal"]
        lines += ["", "## " + r["article_number"] + " · " + p["title"], "",
                  "**原标注**", "", "```json", json.dumps({k: v["value"] for k, v in r["original"].items()}, ensure_ascii=False, indent=2), "```", "",
                  "**发现与理由**", ""]
        lines.extend("- " + f for f in r["findings"])
        lines += ["", "**修正建议**", "", "```json", json.dumps(p, ensure_ascii=False, indent=2), "```", "", "**证据**", ""]
        for source in r["sources"]:
            lines.append(f"- [{source['locator']}]({source['url']})")
        check = r["text_check"]
        if check:
            lines.append(f"- 提供PDF页码：{', '.join(map(str, check['pdf_pages']))}；边界核对：{check['status']}；正文hash：`{r['content_sha256']}`。")
        lines += ["", "**待确认**", ""]
        lines.extend("- " + f for f in r["unresolved"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from pypdf import PdfReader

    for path in (args.output, args.report):
        if path.exists():
            parser.error("输出已存在，请使用新路径: " + str(path))
    raw = args.corpus.read_bytes()
    corpus = json.loads(raw)
    notes_data = json.loads(args.notes.read_text())
    if notes_data["corpus_sha256"] != fingerprint(raw):
        parser.error("建议对应的语料版本已变化")
    notes = notes_data["records"]
    reader = PdfReader(args.pdf)
    result = compare_article_boundaries(corpus, [p.extract_text() or "" for p in reader.pages])
    review = build_review(corpus, notes, result["records"], fingerprint(raw))
    review["text_evidence"] = {"pdf_sha256": fingerprint(args.pdf.read_bytes()), "pages": len(reader.pages),
                               "exact_matched": result["exact_matched"], "statuses": dict(Counter(r["status"] for r in result["records"])),
                               "kind": result["evidence_kind"], "remote_bytes_verified": False}
    for path, content in ((args.output, json.dumps(review, ensure_ascii=False, indent=2) + "\n"), (args.report, render_report(review))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps({"summary": review["summary"], "text_evidence": review["text_evidence"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
