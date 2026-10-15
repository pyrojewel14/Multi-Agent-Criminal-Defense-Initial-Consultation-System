"""显式检查本地评分模型；只有 --download 允许下载，在线评分不调用此入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从仓库根目录或 backend 目录直接运行脚本，不要求安装项目包。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.llm.config import RerankerConfig
from app.infrastructure.llm.rerankers._execution import resolve_local_path


def prepare_model(config: RerankerConfig, *, download: bool = False) -> str:
    """优先返回唯一现有目录；显式允许时才加载下载依赖并获取模型。"""
    try:
        return str(resolve_local_path(config.local_path))
    except FileNotFoundError:
        # 已存在多个配置目录需要人工消除歧义，下载不能替代配置修复。
        path = Path(config.local_path)
        if not download or (path.is_dir() and len(list(path.glob("**/config.json"))) > 1):
            raise
    from modelscope import snapshot_download

    downloaded_path = snapshot_download(model_id=config.model_name, cache_dir=config.cache_dir)
    return str(resolve_local_path(downloaded_path))


def main(argv: list[str] | None = None) -> int:
    """解析显式模型准备请求；缺失本地模型时返回非零状态。"""
    parser = argparse.ArgumentParser(description="检查本地重排模型，使用 --download 显式下载")
    parser.add_argument("--download", action="store_true", help="允许通过 ModelScope 下载缺失模型")
    args = parser.parse_args(argv)
    try:
        print(prepare_model(RerankerConfig.from_env(), download=args.download))
    except Exception as exc:
        print(f"reranker preparation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
