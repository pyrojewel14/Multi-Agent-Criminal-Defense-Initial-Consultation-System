from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LawDataSource(str, Enum):
    """法条候选允许使用的数据来源。"""

    RAG_UNVERIFIED = "rag_unverified"
    LLM_EXTRACTED = "llm_extracted"
    RAG_VERIFIED = "rag_verified"
    JSON_KEYWORD = "json_keyword"


class CoverageSource(str, Enum):
    """覆盖度分析允许返回的来源状态。"""

    NO_LAWS = "no_laws"
    RAG_UNVERIFIED = "rag_unverified"
    LLM_EXTRACTED = "llm_extracted"
    RAG_VERIFIED = "rag_verified"
    JSON_KEYWORD = "json_keyword"
    INVALID_SOURCE = "invalid_source"
    MISSING_REQUIRED_ELEMENTS = "missing_required_elements"


class LawSourceSchema(BaseModel):
    """校验法条候选的数据来源枚举。"""

    model_config = ConfigDict(extra="ignore")

    data_source: LawDataSource


class CoverageCandidateSchema(LawSourceSchema):
    """校验可参与覆盖度计算的法条候选。"""

    required_elements: list[Any] = Field(min_length=1)
