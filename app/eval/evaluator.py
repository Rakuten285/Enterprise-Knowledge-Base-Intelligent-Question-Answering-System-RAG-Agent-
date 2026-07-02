"""
Ragas 评测器：对 RAG 链路做自动化指标评估

评测流程：
  1. 从 eval_dataset.json 读取 50 条 Q&A
  2. 对每条问题跑完整 RAG 链路（检索 + 生成）
  3. 用 Ragas 计算三项指标：Context Recall / Faithfulness / Answer Relevancy
  4. 输出每条得分 + 整体均值

支持参数扫描：
  EvaluatorConfig 里调 chunk_size / top_k，可做参数对比实验
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DATASET_PATH = ROOT / "data" / "eval_dataset.json"
RAW_DIR = ROOT / "data" / "raw"


@dataclass
class EvaluatorConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    llm_provider: str | None = None  # None → 使用 .env 里的 LLM_PROVIDER


@dataclass
class EvalResult:
    config: EvaluatorConfig
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"chunk_size={self.config.chunk_size}, top_k={self.config.top_k} | "
            f"Context Recall={self.context_recall:.4f}  "
            f"Faithfulness={self.faithfulness:.4f}  "
            f"Answer Relevancy={self.answer_relevancy:.4f}  "
            f"({self.elapsed_seconds:.1f}s)"
        )


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_pipeline(config: EvaluatorConfig):
    """构建并返回 (pipeline, hybrid_retriever)，按给定 config 重建索引"""
    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    n = pipeline.build_from_directory(
        RAW_DIR,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    logger.info(f"索引构建完成：{n} 个分块 (chunk_size={config.chunk_size})")
    return pipeline


def run_rag_on_sample(pipeline, query: str, top_k: int, provider: str | None):
    """对单条问题跑检索+生成，返回 (answer_text, retrieved_contexts)"""
    from app.rag.chain import generate_answer

    chunks = pipeline.retrieve(query, k=top_k)
    rag_answer = generate_answer(query, chunks, provider=provider)

    retrieved_contexts = [c.content for c in rag_answer.retrieved_chunks]
    return rag_answer.answer, retrieved_contexts


def evaluate(config: EvaluatorConfig | None = None) -> EvalResult:
    """
    主评测入口：跑完整 50 条评测集，返回 EvalResult

    Args:
        config: 评测参数，None 时使用默认值 (chunk_size=500, top_k=5)
    """
    if config is None:
        config = EvaluatorConfig()

    # 延迟导入 ragas，避免启动时报错（未安装时有明确错误提示）
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as e:
        raise ImportError(
            f"缺少评测依赖：{e}\n"
            "请运行：pip install ragas datasets -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e

    from app.core.llm_factory import get_llm
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    dataset = load_eval_dataset()
    logger.info(f"加载评测集：{len(dataset)} 条问题")

    pipeline = _build_pipeline(config)

    questions, answers, contexts, ground_truths = [], [], [], []

    t0 = time.time()
    for i, sample in enumerate(dataset, 1):
        q = sample["question"]
        gt = sample["ground_truth"]
        logger.info(f"[{i}/{len(dataset)}] 处理：{q[:40]}...")
        try:
            ans, ctxs = run_rag_on_sample(pipeline, q, config.top_k, config.llm_provider)
        except Exception as exc:
            logger.error(f"  第 {i} 条出错：{exc}")
            ans = ""
            ctxs = []
        questions.append(q)
        answers.append(ans)
        contexts.append(ctxs)
        ground_truths.append(gt)

    logger.info("所有问题处理完毕，开始 Ragas 打分...")

    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Ragas 需要一个 LLM 和 Embedding 做评判——复用 DeepSeek + 本地 BGE
    # Ragas 的 LLM wrapper 接受 langchain LLM
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from app.rag.embeddings.bge_local import BGELocalEmbeddings as BGEEmbedding

    ragas_llm = LangchainLLMWrapper(get_llm(provider=config.llm_provider, temperature=0.0))
    ragas_emb = LangchainEmbeddingsWrapper(BGEEmbedding())

    # strictness=1 → 只生成1个问题用于评估；默认值>1时 DeepSeek 会报 n>1 不支持的错误
    metrics = [
        ContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1),
    ]

    result = ragas_evaluate(ragas_dataset, metrics=metrics)
    elapsed = time.time() - t0

    # result.scores 是 Dataset，转 list[dict]
    per_sample = result.scores.to_list() if hasattr(result.scores, "to_list") else []

    cr = float(result["context_recall"])
    fa = float(result["faithfulness"])
    ar = float(result["answer_relevancy"])

    eval_result = EvalResult(
        config=config,
        context_recall=cr,
        faithfulness=fa,
        answer_relevancy=ar,
        per_sample=per_sample,
        elapsed_seconds=elapsed,
    )
    logger.success(f"评测完成：{eval_result.summary()}")
    return eval_result
