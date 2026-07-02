"""
智谱 GLM Embedding Provider
embedding-3 模型，1024/2048 维可选（默认 2048），OpenAI 兼容接口调用。
注册账号有免费额度: https://open.bigmodel.cn
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings.base import BaseEmbeddingProvider


class ZhipuEmbeddingProvider(BaseEmbeddingProvider):
    name = "zhipu"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        if not settings.zhipu_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=zhipu 但未配置 ZHIPU_API_KEY，"
                "请在 .env 中设置，或切换 EMBEDDING_PROVIDER=bge_local 使用免费本地方案"
            )

        self.model = model or settings.zhipu_embedding_model
        logger.info(f"初始化智谱 Embedding: {self.model}")
        self._client = OpenAI(
            api_key=settings.zhipu_api_key, base_url=settings.zhipu_base_url
        )
        self._dimension = 2048

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 智谱 API 单次请求建议不超过一定批量，这里保守按 16 一批
        all_embeddings: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension
