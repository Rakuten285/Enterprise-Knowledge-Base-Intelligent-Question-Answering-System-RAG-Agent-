"""
混合检索模块 (Hybrid Retrieval)
========================================
融合两条召回路径：
  1. 向量检索 (FAISS)   —— 擅长语义相似，能召回同义改写、上下位概念
  2. BM25 关键词检索    —— 擅长精确匹配专有名词、编号、数值等

融合策略支持两种，由 settings.hybrid_fusion_strategy 控制：
  - rrf:      Reciprocal Rank Fusion，只看排名不看原始分数，对两路分数量纲不一致更鲁棒（推荐）
  - weighted: 对两路分数分别归一化后加权求和，权重可调 (VECTOR_WEIGHT / BM25_WEIGHT)

融合之后可选接入 MMR 做结果去重：在候选集中按"与查询相关 + 与已选结果不重复"
两个目标重排，避免返回多个高度相似、信息冗余的分块。
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.retrievers.bm25_retriever import BM25Retriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.document import ChunkMetadata, RetrievedChunk


def _doc_key(doc: Document) -> str:
    """用 source + chunk_index 作为文档去重/对齐的唯一键"""
    meta = doc.metadata
    return f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', id(doc))}"


def _min_max_normalize(scores: list[float]) -> list[float]:
    """归一化到 [0, 1]，用于 weighted 融合策略；全部相同时返回全 1，避免除零"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class FusionCandidate:
    document: Document
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0


class HybridRetriever:
    """
    混合检索器：组合 FAISSVectorStore 与 BM25Retriever，
    对外提供单一的 retrieve() 接口，内部处理召回、融合、去重全流程。
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    # ---------------------------------------------------------------
    # 融合策略
    # ---------------------------------------------------------------
    def _fuse_rrf(
        self,
        vector_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        rrf_k: int,
    ) -> list[FusionCandidate]:
        """
        Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank))
        rank 从 0 开始计数；rrf_k 越大，排名差异的影响越平滑。
        优点：不依赖原始分数的量纲（向量是 L2 距离，BM25 是 TF-IDF 加权分，天然不可比），
        只用排名做融合，工程上更稳健，是业界混合检索的常见默认选择。
        """
        candidates: dict[str, FusionCandidate] = {}

        for rank, (doc, score) in enumerate(vector_results):
            key = _doc_key(doc)
            candidates[key] = FusionCandidate(
                document=doc, vector_rank=rank, vector_score=score
            )

        for rank, (doc, score) in enumerate(bm25_results):
            key = _doc_key(doc)
            if key in candidates:
                candidates[key].bm25_rank = rank
                candidates[key].bm25_score = score
            else:
                candidates[key] = FusionCandidate(
                    document=doc, bm25_rank=rank, bm25_score=score
                )

        for c in candidates.values():
            rrf_score = 0.0
            if c.vector_rank is not None:
                rrf_score += 1.0 / (rrf_k + c.vector_rank + 1)
            if c.bm25_rank is not None:
                rrf_score += 1.0 / (rrf_k + c.bm25_rank + 1)
            c.fused_score = rrf_score

        return sorted(candidates.values(), key=lambda c: c.fused_score, reverse=True)

    def _fuse_weighted(
        self,
        vector_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        vector_weight: float,
        bm25_weight: float,
    ) -> list[FusionCandidate]:
        """
        加权分数融合：分别 min-max 归一化后按权重求和。
        注意 FAISS 默认返回 L2 距离（越小越相似），这里转换为"相似度"口径（越大越相似）
        再归一化，确保与 BM25（越大越相关）方向一致。
        """
        candidates: dict[str, FusionCandidate] = {}

        vector_docs = [d for d, _ in vector_results]
        # L2 距离 -> 相似度：取负数后归一化，距离越小 -> 负值越大 -> 归一化后越接近1
        vector_sim_raw = [-s for _, s in vector_results]
        vector_sim_norm = _min_max_normalize(vector_sim_raw)

        bm25_docs = [d for d, _ in bm25_results]
        bm25_scores_raw = [s for _, s in bm25_results]
        bm25_norm = _min_max_normalize(bm25_scores_raw)

        for rank, (doc, norm_score) in enumerate(zip(vector_docs, vector_sim_norm)):
            key = _doc_key(doc)
            candidates[key] = FusionCandidate(
                document=doc, vector_rank=rank, vector_score=norm_score
            )

        for rank, (doc, norm_score) in enumerate(zip(bm25_docs, bm25_norm)):
            key = _doc_key(doc)
            if key in candidates:
                candidates[key].bm25_rank = rank
                candidates[key].bm25_score = norm_score
            else:
                candidates[key] = FusionCandidate(
                    document=doc, bm25_rank=rank, bm25_score=norm_score
                )

        for c in candidates.values():
            v = c.vector_score or 0.0
            b = c.bm25_score or 0.0
            c.fused_score = vector_weight * v + bm25_weight * b

        return sorted(candidates.values(), key=lambda c: c.fused_score, reverse=True)

    # ---------------------------------------------------------------
    # MMR 去重（在融合候选集上做，而非单一召回路径上）
    # ---------------------------------------------------------------
    def _apply_mmr(
        self,
        query: str,
        candidates: list[FusionCandidate],
        k: int,
        lambda_mult: float,
    ) -> list[FusionCandidate]:
        """
        对融合后的候选集做 MMR 重排，降低返回结果之间的语义冗余。
        复用向量库的 embedding_provider 计算 query 与候选文档的向量，
        以及候选文档之间的相似度，纯 numpy 实现避免再次依赖底层 FAISS 内部结构。
        """
        import numpy as np

        if len(candidates) <= k:
            return candidates

        embedder = self.vector_store.embedding_provider
        doc_texts = [c.document.page_content for c in candidates]
        doc_vectors = np.array(embedder.embed_documents(doc_texts))
        query_vector = np.array(embedder.embed_query(query))

        # 向量已归一化（BGE）或近似归一化，用点积近似 cosine 相似度
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            return float(np.dot(a, b) / denom)

        relevance = [cosine_sim(query_vector, v) for v in doc_vectors]

        selected_idx: list[int] = []
        remaining_idx = list(range(len(candidates)))

        while remaining_idx and len(selected_idx) < k:
            if not selected_idx:
                # 第一个直接选相关性最高的
                best = max(remaining_idx, key=lambda i: relevance[i])
            else:
                def mmr_score(i: int) -> float:
                    max_sim_to_selected = max(
                        cosine_sim(doc_vectors[i], doc_vectors[j]) for j in selected_idx
                    )
                    return lambda_mult * relevance[i] - (1 - lambda_mult) * max_sim_to_selected

                best = max(remaining_idx, key=mmr_score)

            selected_idx.append(best)
            remaining_idx.remove(best)

        return [candidates[i] for i in selected_idx]

    # ---------------------------------------------------------------
    # 对外主入口
    # ---------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        fusion_strategy: str | None = None,
        use_mmr: bool | None = None,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        混合检索主入口。

        流程: 向量召回 top-N + BM25 召回 top-N -> 融合(RRF/加权) -> 截断到候选池
              -> (可选) MMR 重排去重 -> 返回最终 top-k

        Args:
            query: 用户查询
            k: 最终返回结果数，默认读取 settings.top_k
            fusion_strategy: "rrf" | "weighted"，默认读取 settings.hybrid_fusion_strategy
            use_mmr: 是否启用 MMR 去重，默认读取 settings.mmr_enabled
            vector_k / bm25_k: 两路召回各自取多少候选用于融合，默认取 mmr_fetch_k 保证候选池足够大
        """
        k = k or settings.top_k
        fusion_strategy = fusion_strategy or settings.hybrid_fusion_strategy
        use_mmr = settings.mmr_enabled if use_mmr is None else use_mmr
        candidate_pool_size = max(vector_k or settings.mmr_fetch_k, k)

        vector_results = self.vector_store.similarity_search_with_score(
            query, k=candidate_pool_size
        )
        bm25_results = self.bm25_retriever.search(
            query, k=bm25_k or candidate_pool_size
        )

        logger.debug(
            f"混合检索 query='{query[:30]}...' "
            f"向量召回={len(vector_results)} BM25召回={len(bm25_results)} "
            f"策略={fusion_strategy} MMR={use_mmr}"
        )

        if fusion_strategy == "rrf":
            fused = self._fuse_rrf(vector_results, bm25_results, settings.rrf_k)
        elif fusion_strategy == "weighted":
            fused = self._fuse_weighted(
                vector_results,
                bm25_results,
                settings.vector_weight,
                settings.bm25_weight,
            )
        else:
            raise ValueError(f"未知融合策略: {fusion_strategy}，支持 rrf / weighted")

        # 截断候选池，再做 MMR（候选池太大会显著拖慢 MMR 的两两相似度计算）
        pool = fused[: max(candidate_pool_size, k)]

        if use_mmr:
            pool = self._apply_mmr(query, pool, k=k, lambda_mult=settings.mmr_lambda)
        else:
            pool = pool[:k]

        results: list[RetrievedChunk] = []
        for c in pool:
            meta = c.document.metadata
            source_label = "hybrid"
            if c.vector_rank is not None and c.bm25_rank is None:
                source_label = "vector"
            elif c.bm25_rank is not None and c.vector_rank is None:
                source_label = "bm25"

            results.append(
                RetrievedChunk(
                    content=c.document.page_content,
                    metadata=ChunkMetadata(
                        source=meta.get("source", "unknown"),
                        file_type=meta.get("file_type", "unknown"),
                        chunk_index=meta.get("chunk_index", -1),
                        page_number=meta.get("page"),
                        total_chunks=meta.get("total_chunks"),
                    ),
                    score=c.fused_score,
                    retrieval_source=source_label,
                )
            )

        return results
