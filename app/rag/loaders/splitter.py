"""
文档分块模块
使用 RecursiveCharacterTextSplitter，按 段落 -> 句子 -> 词 的优先级递归切分，
中文场景下额外加入中文标点作为分隔符，避免句子被从中间切断。
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger

# 中文优先分隔符：段落 > 中文句号/换行 > 中文逗号 > 英文标点 > 字符
CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """构建分块器，chunk_size/overlap 默认读取全局配置，便于评测阶段做参数扫描"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    对文档列表分块，并在 metadata 中写入 chunk_index / total_chunks，
    保留原始 source / file_type / page 等信息，便于检索结果追溯来源。
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # 按 source 分组重新编号 chunk_index，方便定位"某文档第几块"
    source_counters: dict[str, int] = {}
    source_totals: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_totals[src] = source_totals.get(src, 0) + 1

    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["total_chunks"] = source_totals[src]
        source_counters[src] = idx + 1

    logger.info(
        f"分块完成: {len(documents)} 个原始片段 -> {len(chunks)} 个分块 "
        f"(chunk_size={chunk_size or settings.chunk_size}, "
        f"overlap={chunk_overlap or settings.chunk_overlap})"
    )
    return chunks
