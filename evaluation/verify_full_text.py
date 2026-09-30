"""对提供的政府 PDF 做正文子串比对，输出程序证据而非法律审查结论。"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def compare_text(corpus: dict, text: str) -> dict:
    """移除空白和页码后检查正文，保留未命中条号供人工核对。"""
    normalized = re.sub(r"\s+", "", re.sub(r"－\s*\d+\s*－", "", text))
    articles = [a for c in corpus["chapters"] for a in c["articles"]]
    missing = [
        a["article_number"]
        for a in articles
        if re.sub(r"\s+", "", a["content"]) not in normalized
    ]
    return {
        "records": len(articles),
        "content_matched": len(articles) - len(missing),
        "unmatched": missing,
        "legal_review_completed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-refresh-http-status", type=int)
    args = parser.parse_args()
    from pypdf import PdfReader

    raw = args.corpus.read_bytes()
    corpus = json.loads(raw)
    reader = PdfReader(args.pdf)
    result = compare_text(
        corpus, "\n".join(page.extract_text() or "" for page in reader.pages)
    )
    result.update(
        {
            "evidence_kind": "supplied-pdf-programmatic-substring-check",
            "official_url": corpus["metadata"]["official_text"]["url"],
            "pdf_sha256": "sha256:" + hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            "corpus_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "pages": len(reader.pages),
            "source_refresh_http_status": args.source_refresh_http_status,
            "limitations": "使用提供的PDF，未验证当前远端PDF字节hash。子串核验不代表逐条法律人工审查。",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result["unmatched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
