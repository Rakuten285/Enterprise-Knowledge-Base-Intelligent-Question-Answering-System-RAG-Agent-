"""
建库脚本：解析 data/raw/ 下所有文档，分块后构建 FAISS + BM25 索引并持久化到 data/vectorstore/

用法:
    python scripts/build_index.py
    python scripts/build_index.py --raw-dir data/raw --chunk-size 500 --chunk-overlap 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import logger
from app.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="构建企业知识库 RAG 索引")
    parser.add_argument("--raw-dir", default="data/raw", help="原始文档目录")
    parser.add_argument("--chunk-size", type=int, default=None, help="分块大小，默认读取 .env")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="分块重叠，默认读取 .env")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.error(
            f"目录 {raw_dir} 不存在或为空，请放入 PDF/Word/TXT/Markdown 文档后重试"
        )
        sys.exit(1)

    t0 = time.time()
    pipeline = RAGPipeline()
    n_chunks = pipeline.build_from_directory(
        raw_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    elapsed = time.time() - t0

    logger.success(
        f"[OK] 建库完成: {n_chunks} 个分块, 耗时 {elapsed:.1f}s, "
        f"索引已保存至 {pipeline.vector_store.index_path}"
    )


if __name__ == "__main__":
    main()
