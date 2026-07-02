"""
Embedding Provider 工厂
通过 EMBEDDING_PROVIDER 环境变量决定使用哪种实现，下游代码统一调用 get_embedding_provider()，
切换 Provider 时无需改动任何业务代码。
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.rag.embeddings.base import BaseEmbeddingProvider


@lru_cache
def get_embedding_provider(provider: str | None = None) -> BaseEmbeddingProvider:
    """
    获取 Embedding Provider 实例（带缓存，避免重复加载模型/重复建连接）

    Args:
        provider: 显式指定 provider 名称（bge_local / openai / zhipu），
                  默认读取 settings.embedding_provider
    """
    selected = provider or settings.embedding_provider

    if selected == "bge_local":
        from app.rag.embeddings.bge_local import BGELocalEmbeddings

        return BGELocalEmbeddings()
    elif selected == "openai":
        from app.rag.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider()
    elif selected == "zhipu":
        from app.rag.embeddings.zhipu_provider import ZhipuEmbeddingProvider

        return ZhipuEmbeddingProvider()
    else:
        raise ValueError(
            f"未知的 EMBEDDING_PROVIDER: {selected}，支持: bge_local / openai / zhipu"
        )
