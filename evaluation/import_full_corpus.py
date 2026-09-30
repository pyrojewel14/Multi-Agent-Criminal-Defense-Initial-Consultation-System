"""将候选语料隔离为公开正文资产与待复核标注；不覆盖现有版本。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.knowledge.full_law_corpus import prepare_full_corpus, review_queue, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.review_queue.exists():
        raise FileExistsError("输出已存在，请使用新版本路径")
    raw = args.candidate.read_bytes()
    corpus = prepare_full_corpus(json.loads(raw), sha256(raw))
    for path, data in [
        (args.output, corpus),
        (args.review_queue, review_queue(corpus)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "version": corpus["metadata"]["dataset_version"],
                "records": 505,
                "legal_review_completed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
