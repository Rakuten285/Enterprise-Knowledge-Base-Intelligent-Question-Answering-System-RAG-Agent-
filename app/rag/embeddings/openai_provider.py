"""
OpenAI Embedding Provider
默认使用 text-embedding-3-small（1536 维），需配置 OPENAI_API_KEY。
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider

# text-embedding-3-small / large 已知维度，避免每次都发请求探测
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai 但未配置 OPENAI_API_KEY，"
                "请在 .env 中设置，或切换 EMBEDDING_PROVIDER=bge_local 使用免费本地方案"
            )

        self.model = model or settings.openai_embedding_model
        logger.info(f"初始化 OpenAI Embedding: {self.model}")
        self._client = OpenAIEmbeddings(
            model=self.model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
        self._dimension = _KNOWN_DIMENSIONS.get(self.model, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension
