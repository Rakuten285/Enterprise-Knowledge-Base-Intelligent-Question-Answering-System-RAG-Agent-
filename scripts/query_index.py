"""
检索测试脚本：加载已构建的索引，对单条查询执行混合检索并打印结果

用法:
    python scripts/query_index.py "公司的年假制度是怎样的？"
    python scripts/query_index.py "报销流程" --strategy weighted --no-mmr --k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import logger
from app.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="测试混合检索效果")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--k", type=int, default=None, help="返回结果数")
    parser.add_argument(
        "--strategy", choices=["rrf", "weighted"], default=None, help="融合策略"
    )
    parser.add_argument("--no-mmr", action="store_true", help="禁用 MMR 去重")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.load()

    results = pipeline.retrieve(
        args.query,
        k=args.k,
        fusion_strategy=args.strategy,
        use_mmr=not args.no_mmr,
    )

    print(f"\n查询: {args.query}")
    print(f"返回 {len(results)} 条结果\n" + "=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] 来源: {r.metadata.source} (chunk #{r.metadata.chunk_index})"
              f" | 召回路径: {r.retrieval_source} | 融合分: {r.score:.4f}")
        preview = r.content[:200].replace("\n", " ")
        print(f"    {preview}{'...' if len(r.content) > 200 else ''}")


if __name__ == "__main__":
    main()
