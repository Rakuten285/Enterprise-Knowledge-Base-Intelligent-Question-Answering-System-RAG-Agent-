"""
完整 RAG 问答脚本：加载索引 → 混合检索 → LLM 生成 → 打印答案与来源

用法:
    python scripts/rag_query.py "公司年假有多少天"
    python scripts/rag_query.py "报销流程" --k 3 --provider deepseek
    python scripts/rag_query.py "密码规则" --no-mmr --strategy weighted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import logger
from app.rag.chain import generate_answer
from app.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 完整问答（检索 + 生成）")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--k", type=int, default=None, help="检索返回 top-k 条，默认读 .env TOP_K")
    parser.add_argument("--provider", default=None, help="LLM provider: deepseek/zhipu/qwen/openai")
    parser.add_argument("--strategy", choices=["rrf", "weighted"], default=None)
    parser.add_argument("--no-mmr", action="store_true", help="禁用 MMR 去重")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.load()

    chunks = pipeline.retrieve(
        args.query,
        k=args.k,
        fusion_strategy=args.strategy,
        use_mmr=not args.no_mmr,
    )

    result = generate_answer(
        query=args.query,
        chunks=chunks,
        provider=args.provider,
    )

    print("\n" + "=" * 60)
    print(f"问题：{args.query}")
    print("=" * 60)
    print(f"\n【回答】\n{result.answer}")
    print(f"\n【参考来源】{', '.join(result.sources)}")
    print("\n【检索到的原文片段】")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        preview = chunk.content[:150].replace("\n", " ")
        print(f"  [{i}] {chunk.metadata.source} chunk#{chunk.metadata.chunk_index}"
              f" | 路径:{chunk.retrieval_source} 分:{chunk.score:.4f}")
        print(f"      {preview}{'...' if len(chunk.content) > 150 else ''}")


if __name__ == "__main__":
    main()
