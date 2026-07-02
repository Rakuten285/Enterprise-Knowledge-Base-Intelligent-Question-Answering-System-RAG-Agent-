"""
BM25 关键词检索模块
向量检索擅长语义相似但对精确关键词（专有名词、编号、数值）召回较弱，
BM25 基于词频统计正好互补。中文场景下需先分词，这里用 jieba。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import logger

# 常见中文停用词，过滤后能让 BM25 更聚焦实质性关键词
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "他", "这", "中", "大",
    "为", "上", "个", "国", "到", "以", "说", "时", "要", "就", "出",
    "会", "可", "也", "你", "对", "生", "能", "自", "之", "等", "与",
    "及", "或", "而", "之类", "一个", "什么", "如何", "怎么", "吗", "呢",
}


def tokenize(text: str) -> list[str]:
    """中文分词 + 去停用词 + 去空白"""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]


class BM25Retriever:
    """基于 rank_bm25 的关键词检索器，支持持久化以避免每次启动重新分词建索引"""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def index_path(self) -> Path:
        return settings.vectorstore_path / f"{settings.vectorstore_index_name}_bm25.pkl"

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("文档列表为空，无法构建 BM25 索引")

        logger.info(f"开始构建 BM25 索引: {len(documents)} 个分块")
        self._documents = documents
        self._tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 索引构建完成")

    def save(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，无法保存")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {"documents": self._documents, "tokenized_corpus": self._tokenized_corpus},
                f,
            )
        logger.info(f"BM25 索引已持久化到: {self.index_path}")

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"未找到已持久化的 BM25 索引: {self.index_path}")
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 索引加载完成: {len(self._documents)} 个分块")

    def load_or_build(self, documents: list[Document]) -> None:
        try:
            self.load()
        except FileNotFoundError:
            self.build(documents)
            self.save()

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """返回 (Document, BM25得分) 列表，按得分降序；得分越高越相关"""
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未初始化，请先调用 build() / load()")

        k = k or settings.top_k
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self._documents[i], float(scores[i])) for i in ranked_indices]
