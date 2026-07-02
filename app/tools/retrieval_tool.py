"""
知识库检索工具：封装阶段1的 RAGPipeline + generate_answer，暴露为 LangChain @tool

注意：不维护独立的 _pipeline 单例。通过 get_shared_pipeline() 与 chat.py 共享同一个实例，
确保 /kb/build 重建后 Agent 模式也能立即使用新索引。
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.rag.chain import generate_answer


def _get_pipeline():
    """从 chat 模块获取共享 pipeline 单例，避免两份独立实例导致 /kb/build 后索引不同步"""
    import app.api.routes.chat as chat_module
    return chat_module._get_pipeline()


@tool
def search_knowledge_base(query: str) -> str:
    """从企业内部知识库检索信息。
    适用于：员工手册、年假制度、考勤规定、薪酬福利、离职流程、IT安全规范、采购管理制度等内部文档。
    不适用于：需要实时数据、员工个人信息、数学计算等场景。
    """
    pipeline = _get_pipeline()
    chunks = pipeline.retrieve(query)
    if not chunks:
        return "知识库中未找到与该问题相关的内容。"
    result = generate_answer(query=query, chunks=chunks)
    sources = "、".join(result.sources)
    return f"{result.answer}\n\n[来源：{sources}]"
