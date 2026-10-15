"""按通道排名等权融合，并独立保存各通道原始证据。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from .models import CandidateOrigin, ChannelResult, RetrievalCandidate


def document_key(candidate: RetrievalCandidate) -> str:
    """用来源与存储行身份去重；无行 ID 的领域候选保留其自身身份。"""
    return json.dumps(
        [candidate.source_id, candidate.metadata.get("row_id", candidate.candidate_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def choose_stable_representative(candidates: Sequence[RetrievalCandidate]) -> RetrievalCandidate:
    """通用文档按稳定身份选择代表，领域服务可注入自己的资格规则。"""
    return min(candidates, key=lambda candidate: (candidate.source_id, candidate.candidate_id))


def rrf_fuse(
    channels: Mapping[str, ChannelResult],
    *,
    key: Callable[[RetrievalCandidate], str],
    choose_representative: Callable[[Sequence[RetrievalCandidate]], RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    """每个通道每个身份只加一次 1/(60+rank)，不以原始分数或资格加权。"""
    groups: dict[str, list[RetrievalCandidate]] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    origins: dict[str, list[CandidateOrigin]] = {}

    # 原始证据全部保留；同一身份在同一通道重复出现时，仅首个排名贡献分数。
    for name, channel in channels.items():
        seen = set()
        for rank, candidate in enumerate(channel.candidates, 1):
            identity = key(candidate)
            groups.setdefault(identity, []).append(candidate)
            evidence = candidate.origins or (
                CandidateOrigin(candidate.source_id, candidate.candidate_id, name, candidate.metadata),
            )
            history = origins.setdefault(identity, [])
            for origin in evidence:
                if origin not in history:
                    history.append(origin)
            if identity in seen:
                continue
            seen.add(identity)
            ranks.setdefault(identity, {})[name] = rank
            if candidate.candidate_id in channel.scores:
                scores.setdefault(identity, {})[name] = channel.scores[candidate.candidate_id]

    # 正文代表由调用方选择，各通道排名独立参与 RRF，不混加距离或关键词分数。
    fused = []
    for identity, candidates in groups.items():
        representative = choose_representative(tuple(candidates))
        score = sum(1.0 / (60 + rank) for rank in ranks[identity].values())
        fused.append(
            (
                identity,
                replace(
                    representative,
                    rank=None,
                    fusion_score=score,
                    recall_ranks=ranks[identity],
                    recall_scores=scores.get(identity, {}),
                    origins=tuple(origins[identity]),
                ),
            )
        )
    # 融合分数相同时按稳定身份排序，避免结果依赖映射遍历顺序。
    fused.sort(key=lambda item: (-item[1].fusion_score, item[0]))
    return tuple(candidate for _, candidate in fused)
