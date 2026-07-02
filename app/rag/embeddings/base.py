"""
Embedding Provider 抽象基类
统一接口，遵循 LangChain Embeddings 协议 (embed_documents / embed_query)，
方便在不同向量存储 / 检索组件之间直接替换，无需改动下游代码。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class BaseEmbeddingProvider(Embeddings, ABC):
    """所有 Embedding 实现的统一基类"""

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度，FAISS 建库时需要"""
        ...
