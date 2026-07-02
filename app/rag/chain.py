"""
RAG 生成链：检索 → Prompt 拼装 → LLM 生成 → 带来源的答案输出

设计原则：
- chain.py 只负责"把检索结果和 query 拼成 Prompt、调 LLM 拿到回答"这一层
- 不做检索，不做分块——这些在 pipeline.py 里；chain.py 接受 RetrievedChunk 列表
- Prompt 模板写在本文件里，方便后续做 Prompt 版本化管理（阶段4目标）
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm
from app.core.logging import logger
from app.schemas.document import RetrievedChunk

# --------------------------------------------------------------------------
# Prompt 模板
# --------------------------------------------------------------------------
_SYSTEM_PROMPT = """你是一个企业内部知识库智能助手。请根据以下检索到的参考文档回答用户问题。

要求：
1. 只基于给定的参考文档作答，不要使用文档之外的知识
2. 如果参考文档中没有足够的信息回答问题，明确告知用户"文档中未找到相关信息"，不要编造
3. 回答要简洁准确，可以直接引用文档中的关键数字、规定和表述
4. 引用内容时注明来源文件名（如：根据《员工手册》…）"""

_CONTEXT_TEMPLATE = """参考文档（共 {n} 条，按相关度排序）：

{context}

---
用户问题：{query}"""


def _extract_cited_sources(answer: str, all_sources: list[str]) -> list[str]:
    """从答案文本中找出实际被引用的来源文件名。
    匹配：《员工手册》《IT安全规范》或文件名去掉扩展名后的字符串。
    返回空列表时调用方兜底使用全部来源。
    """
    cited = []
    for src in all_sources:
        # 去掉扩展名作为匹配关键词（如 "员工手册.md" → "员工手册"）
        stem = re.sub(r'\.\w+$', '', src)
        if stem in answer:
            cited.append(src)
    return cited


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.source
        parts.append(f"[{i}] 来源：{source}\n{chunk.content}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 核心数据结构
# --------------------------------------------------------------------------
@dataclass
class RAGAnswer:
    """RAG 完整回答，包含生成文本和引用来源"""
    answer: str
    sources: list[str]       # 去重后的来源文件名列表
    retrieved_chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------
# 核心函数
# --------------------------------------------------------------------------
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    provider: str | None = None,
    temperature: float = 0.1,
) -> RAGAnswer:
    """
    给定 query 和已检索的 chunks，调 LLM 生成回答。

    Args:
        query: 用户原始问题
        chunks: 已经过检索+排序的文档片段列表
        provider: LLM provider，None 时使用 .env 里的 LLM_PROVIDER
        temperature: 生成温度，RAG 场景建议低温减少幻觉
    """
    if not chunks:
        logger.warning("generate_answer: 检索结果为空，无法生成有依据的回答")
        return RAGAnswer(
            answer="抱歉，未在知识库中检索到与您问题相关的文档内容。",
            sources=[],
            retrieved_chunks=[],
        )

    llm = get_llm(provider=provider, temperature=temperature)

    context_str = _format_context(chunks)
    user_content = _CONTEXT_TEMPLATE.format(
        n=len(chunks),
        context=context_str,
        query=query,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    logger.info(f"调用 LLM 生成回答，context 包含 {len(chunks)} 个分块")
    response = llm.invoke(messages)
    answer_text = response.content

    # sources 三级策略：
    # 1. 拒答（未找到相关信息）→ 空列表，不误导用户
    # 2. 答案里有显式文档引用 → 只列被引用的文档
    # 3. 有答案但无显式引用 → 列出所有检索来源
    all_sources = list(dict.fromkeys(c.metadata.source for c in chunks))
    _REFUSAL_PHRASES = ("未找到相关信息", "没有找到相关", "无法回答", "not found in")
    is_refusal = any(p in answer_text for p in _REFUSAL_PHRASES)
    if is_refusal:
        sources = []
    else:
        cited = _extract_cited_sources(answer_text, all_sources)
        sources = cited if cited else all_sources

    return RAGAnswer(
        answer=answer_text,
        sources=sources,
        retrieved_chunks=chunks,
    )
