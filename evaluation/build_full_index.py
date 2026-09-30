"""用实际 Ollama 模型构建独立全量公共索引。"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "backend/data/law_knowledge/criminal_law_full.json",
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--collection", default="criminal_law_full")
    args = parser.parse_args()
    import httpx
    from dotenv import load_dotenv

    load_dotenv(ROOT / "backend/.env", override=False)
    if os.getenv("EMBED_MODEL_TYPE", "OLLAMA") != "OLLAMA":
        raise ValueError("本构建入口需要 Ollama embedding")
    model = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    with httpx.Client(trust_env=False, timeout=10) as client:
        response = client.get(base + "/api/tags")
        response.raise_for_status()
        models = response.json()["models"]
    digest = next((m["digest"] for m in models if m["name"] == model), None)
    if not digest:
        raise ValueError("embedding model unavailable")

    # 直连本地模型，避免代理环境把 loopback 请求转发到外部。
    def embed(texts):
        with httpx.Client(trust_env=False, timeout=180) as client:
            response = client.post(
                base + "/api/embed",
                json={"model": model, "input": texts, "truncate": False},
            )
            response.raise_for_status()
            return response.json()["embeddings"]

    from app.knowledge.full_law_index import build_full_index

    manifest = build_full_index(
        args.corpus, args.index_dir, args.collection, model, digest, embed
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
