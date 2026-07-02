"""统一的数据结构定义"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """每个分块附带的元信息，便于检索后追溯来源"""

    source: str = Field(..., description="原始文件路径或文件名")
    file_type: str = Field(..., description="pdf / docx / txt / md")
    chunk_index: int = Field(..., description="该分块在文档内的序号")
    page_number: Optional[int] = Field(default=None, description="PDF 页码，若适用")
    total_chunks: Optional[int] = Field(default=None, description="文档总分块数")
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """检索结果统一结构"""

    content: str
    metadata: ChunkMetadata
    score: float = Field(..., description="检索得分；不同召回策略分数口径不同")
    retrieval_source: str = Field(
        default="hybrid", description="vector / bm25 / hybrid，标明该结果来自哪条召回路径"
    )

    class Config:
        arbitrary_types_allowed = True
