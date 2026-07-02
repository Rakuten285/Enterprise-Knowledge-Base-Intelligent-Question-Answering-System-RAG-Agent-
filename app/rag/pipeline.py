"""
RAG Pipeline 统一入口
封装 "建库" 与 "检索" 两大流程，供 scripts/ 与 app/api/ 复用，
避免在多处重复拼装 loader -> splitter -> vectorstore -> retriever 的逻辑。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.core.logging import logger
from app.rag.loaders.document_loader import load_documents_from_dir
from app.rag.loaders.splitter import split_documents
from app.rag.retrievers.bm25_retriever import BM25Retriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.document import RetrievedChunk


class RAGPipeline:
    """RAG 核心链路的统一封装：建库 / 加载 / 检索"""

    def __init__(self) -> None:
        self.vector_store = FAISSVectorStore()
        self.bm25_retriever = BM25Retriever()
        self._hybrid: HybridRetriever | None = None

    # ------------------------------------------------------------------
    # 建库
    # ------------------------------------------------------------------
    def build_from_directory(
        self,
        raw_dir: str | Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """从原始文档目录构建完整索引（向量库 + BM25），返回分块总数"""
        raw_docs = load_documents_from_dir(raw_dir)
        if not raw_docs:
            raise ValueError(f"目录 {raw_dir} 下没有解析出任何有效文档")
        return self.build_from_documents(raw_docs, chunk_size, chunk_overlap)

    def build_from_documents(
        self,
        raw_docs: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        chunks = split_documents(raw_docs, chunk_size, chunk_overlap)

        self.vector_store.build(chunks)
        self.vector_store.save()

        self.bm25_retriever.build(chunks)
        self.bm25_retriever.save()

        self._hybrid = HybridRetriever(self.vector_store, self.bm25_retriever)
        logger.info(f"RAG 索引构建完成，共 {len(chunks)} 个分块")
        return len(chunks)

    # ------------------------------------------------------------------
    # 加载已有索引
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载已持久化的向量库 + BM25 索引"""
        self.vector_store.load()
        self.bm25_retriever.load()
        self._hybrid = HybridRetriever(self.vector_store, self.bm25_retriever)
        logger.info("RAG 索引加载完成（向量库 + BM25）")

    def ensure_loaded(self) -> None:
        if self._hybrid is None:
            self.load()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        fusion_strategy: str | None = None,
        use_mmr: bool | None = None,
    ) -> list[RetrievedChunk]:
        self.ensure_loaded()
        assert self._hybrid is not None
        return self._hybrid.retrieve(
            query, k=k, fusion_strategy=fusion_strategy, use_mmr=use_mmr
        )
