"""
FAISS 向量存储封装
负责：从分块文档构建索引、落盘持久化、重新加载、增量添加文档。
底层复用 langchain_community.vectorstores.FAISS，按统一 Embedding Provider 接口接入，
因此切换 bge_local / openai / zhipu 不影响本模块代码。
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings import get_embedding_provider
from app.rag.embeddings.base import BaseEmbeddingProvider


class FAISSVectorStore:
    """对 LangChain FAISS 的轻量封装，提供建库/加载/持久化/检索的统一入口"""

    def __init__(self, embedding_provider: BaseEmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self._store: FAISS | None = None

    @property
    def index_path(self) -> Path:
        return settings.vectorstore_path / settings.vectorstore_index_name

    @property
    def is_loaded(self) -> bool:
        return self._store is not None

    def build(self, documents: list[Document]) -> FAISS:
        """从分块后的文档列表全新构建索引（覆盖式）"""
        if not documents:
            raise ValueError("文档列表为空，无法构建向量库")

        logger.info(
            f"开始构建 FAISS 索引: {len(documents)} 个分块, "
            f"embedding_provider={self.embedding_provider.name}"
        )
        self._store = FAISS.from_documents(documents, self.embedding_provider)
        logger.info("FAISS 索引构建完成")
        return self._store

    def add_documents(self, documents: list[Document]) -> None:
        """增量添加文档到已有索引（支持知识库追加上传新文件的场景）"""
        if self._store is None:
            logger.warning("索引尚未构建/加载，将基于新文档创建索引")
            self.build(documents)
            return
        logger.info(f"增量添加 {len(documents)} 个分块到现有索引")
        self._store.add_documents(documents)

    def save(self) -> None:
        if self._store is None:
            raise RuntimeError("索引未构建，无法保存")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_path))
        logger.info(f"FAISS 索引已持久化到: {self.index_path}")

    def load(self) -> FAISS:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"未找到已持久化的索引: {self.index_path}，请先运行建库脚本 "
                "(scripts/build_index.py)"
            )
        logger.info(f"加载已持久化的 FAISS 索引: {self.index_path}")
        self._store = FAISS.load_local(
            str(self.index_path),
            self.embedding_provider,
            allow_dangerous_deserialization=True,  # 本地自建索引，可信任
        )
        return self._store

    def load_or_build(self, documents: list[Document]) -> FAISS:
        """优先加载已有索引，不存在则用传入文档新建"""
        try:
            return self.load()
        except FileNotFoundError:
            store = self.build(documents)
            self.save()
            return store

    @property
    def store(self) -> FAISS:
        if self._store is None:
            raise RuntimeError("向量库未初始化，请先调用 build() / load() / load_or_build()")
        return self._store

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """原始向量相似度检索，返回 (Document, 距离分数) —— FAISS 默认是 L2 距离，越小越相似"""
        k = k or settings.top_k
        return self.store.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
    ) -> list[Document]:
        """
        MMR (Maximal Marginal Relevance) 检索，兼顾相关性与多样性，降低冗余召回。

        Args:
            k: 最终返回结果数
            fetch_k: 先取相似度最高的 fetch_k 个候选，再在其中做 MMR 重排去重
            lambda_mult: 相关性权重，1.0=只看相关性(等同普通检索)，0.0=只看多样性
        """
        k = k or settings.top_k
        fetch_k = fetch_k or settings.mmr_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.mmr_lambda
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
