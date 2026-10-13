"""生成绑定原语料的复核旁表；程序核对和代理建议均不放行运行时标注。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ARTICLE = re.compile(r"^\s*(第[一二三四五六七八九十百千零]+条(?:之[一二三四五六七八九十]+)?)\s+(.+)$")
HEADING = re.compile(r"^第[一二三四五六七八九十]+[编章节]\s")
FIELDS = {"title", "charges", "elements", "penalty"}
ROOT = Path(__file__).resolve().parents[1]


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
    text_index = {}
    for record in text_records:
        number = record["article_number"]
        if number not in index or number in text_index:
            raise ValueError("文本检查条号未知或重复: " + number)
        if record["content_sha256"] != fingerprint(index[number]["content"].encode()):
            raise ValueError("文本检查绑定的正文已变化: " + number)
        if record["status"] not in {"exact_match", "mismatch", "missing_anchor", "duplicate_anchor"}:
            raise ValueError("文本检查状态未知: " + number)
        if record["status"] in {"exact_match", "mismatch"} and (
            not record["pdf_pages"] or any(type(p) is not int or p < 1 for p in record["pdf_pages"])
        ):
            raise ValueError("文本检查缺少有效页码: " + number)
        if record["status"] == "exact_match" and record["normalized_pdf_sha256"] != fingerprint(compact(index[number]["content"]).encode()):
            raise ValueError("文本精确匹配指纹不一致: " + number)
        text_index[number] = copy.deepcopy(record)
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


def audit_annotations(corpus_path: Path, demo_path: Path, case_paths: list[Path]) -> dict:
    """只读核对运行资产与固定案例，演示资格不升级为专业复核资格。"""
    # 独立脚本与模块入口共用后端契约，不依赖服务启动或模型配置。
    if str(ROOT / "backend") not in sys.path:
        sys.path.insert(0, str(ROOT / "backend"))
    from app.knowledge.demo_law_annotations import apply_demo_annotations
    from app.knowledge.full_law_corpus import validate_full_corpus
    from app.knowledge.law_knowledge import LAW_KNOWLEDGE_PATH, _normalize_article_number, is_lawref_eligible

    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    validate_full_corpus(corpus)
    corpus["metadata"]["corpus_sha256"] = fingerprint(raw)
    demo_raw = demo_path.read_bytes()
    demo_asset = json.loads(demo_raw)
    runtime = apply_demo_annotations(corpus, demo_path)
    if demo_path.read_bytes() != demo_raw:
        raise ValueError("审计期间演示资产已变化")
    demo_index = {_normalize_article_number(r["article_number"]): r for r in demo_asset["records"]}
    records = []
    for article in articles(runtime):
        number = _normalize_article_number(article["article_number"])
        demo_record = demo_index.get(number)
        qualification = "demo_ready" if demo_record else (
            "project_regression" if article["annotation_layer"] == "project-maintained-v1" else "pending")
        records.append({
            "article_id": number, "article_number": article["article_number"], "status": article["status"],
            "qualification": qualification, "content_sha256": article["text_provenance"]["content_sha256"],
            "annotation_sha256": fingerprint(json.dumps(demo_record or article["annotations"], ensure_ascii=False, sort_keys=True).encode()),
            "annotation_version": demo_asset["annotation_id"] if demo_record else article["annotation_source"],
            "annotation_reviews": {name: {key: annotation[key] for key in (
                "review_status", "legal_review_status", "source", "reviewed_by")}
                for name, annotation in article["annotations"].items()},
            "text_provenance": copy.deepcopy(article["text_provenance"]),
            "sources": copy.deepcopy(demo_record["sources"] if demo_record else [{
                "url": article["text_provenance"]["url"], "locator": article["article_number"]}]),
            "legal_review_status": "pending", "professional_coverage_eligible": False,
            "demo_coverage_eligible": is_lawref_eligible(article),
        })
    index = {r["article_id"]: r for r in records}
    priorities: dict[str, dict] = {}
    case_assets = []
    for path in case_paths:
        case_raw = path.read_bytes()
        cases = [json.loads(line) for line in case_raw.decode("utf-8").splitlines() if line.strip()]
        try:
            label = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            label = path.name
        case_assets.append({"path": label, "sha256": fingerprint(case_raw), "cases": len(cases)})
        seen = set()
        for case in cases:
            number = _normalize_article_number(case["article_id"])
            if number not in index or not isinstance(case["id"], str) or not case["id"].strip() or case["id"] in seen:
                raise ValueError("固定案例条号未知或编号无效: " + label)
            seen.add(case["id"])
            priority = priorities.setdefault(number, {
                "article_id": number, "case_ids": [], "case_sources": [], "qualification": index[number]["qualification"],
                "review_action": "复核现有标注" if index[number]["demo_coverage_eligible"] else "核对标注缺口或非独立罪名边界；不得自动放行",
                "legal_review_status": "pending", "professional_coverage_eligible": False,
            })
            if case["id"] not in priority["case_ids"]:
                priority["case_ids"].append(case["id"])
            if label not in priority["case_sources"]:
                priority["case_sources"].append(label)
    counts = Counter(r["qualification"] for r in records)
    return {
        "schema_version": "annotation-qualification-audit-v1", "corpus_sha256": fingerprint(raw),
        "dataset_version": corpus["metadata"]["dataset_version"], "demo_asset_sha256": fingerprint(demo_raw),
        "demo_annotation_version": demo_asset["annotation_id"], "snapshot_sha256": fingerprint(LAW_KNOWLEDGE_PATH.read_bytes()),
        "official_text": copy.deepcopy(corpus["metadata"]["official_text"]),
        "legal_review_completed": False, "professional_reviewer": None,
        "limitations": "只核对本地资产契约；未核实远端来源当前字节或现行法，不认定个案适用、事实支持或法律结论。专业复核负责人待指定。",
        "summary": {"records": len(records), "active_texts": sum(r["status"] == "active" for r in records),
                    "project_regression": counts["project_regression"], "demo_ready": counts["demo_ready"], "pending": counts["pending"],
                    "demo_coverage_eligible": sum(r["demo_coverage_eligible"] for r in records), "professional_coverage_eligible": 0},
        "case_assets": case_assets, "priority_batch": list(priorities.values()), "records": records,
    }


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
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--audit-only", action="store_true", help="只读审计当前标注资格，不生成或放行标注")
    parser.add_argument("--demo", type=Path)
    parser.add_argument("--cases", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.audit_only:
        if not args.demo or not args.cases or any((args.notes, args.pdf, args.report)):
            parser.error("资格审计需要 --demo 和 --cases，不接受 --notes/--pdf/--report")
        if args.output.exists():
            parser.error("输出已存在，请使用新路径: " + str(args.output))
        result = audit_annotations(args.corpus, args.demo, args.cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 排他创建，防止两个并发审计覆盖同一证据文件。
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps(result["summary"], ensure_ascii=False))
        return
    if not all((args.notes, args.pdf, args.report)) or args.demo or args.cases:
        parser.error("复核旁表需要 --notes/--pdf/--report；资格审计使用 --audit-only")
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
