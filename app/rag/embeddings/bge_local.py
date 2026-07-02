"""
本地 BGE Embedding Provider
基于 sentence-transformers 加载 BAAI/bge-small-zh-v1.5，完全免费、离线运行，
无需任何 API Key，是本项目的默认 Embedding 方案。

注意：BGE 官方建议查询文本加指令前缀 "为这个句子生成表示以用于检索相关文章："
以提升检索效果，文档侧不需要加。
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGELocalEmbeddings(BaseEmbeddingProvider):
    """本地 BGE 中文 Embedding，模型首次调用时自动从 HuggingFace 下载并缓存"""

    name = "bge_local"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.bge_model_name
        self.device = device or settings.bge_device

        logger.info(f"加载本地 BGE Embedding 模型: {self.model_name} (device={self.device})")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"BGE 模型加载完成，向量维度={self._dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # 归一化后用内积等价于 cosine 相似度
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # 检索查询加指令前缀，是 BGE 官方推荐的非对称检索增强技巧
        prefixed = QUERY_INSTRUCTION + text
        embedding = self._model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


@lru_cache
def get_bge_embeddings() -> BGELocalEmbeddings:
    """缓存模型实例，避免重复加载（加载耗时几秒到几十秒）"""
    return BGELocalEmbeddings()
