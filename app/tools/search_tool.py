"""
网络搜索工具：DuckDuckGo，无需 API Key。
依赖 duckduckgo-search，未安装时优雅降级返回提示。
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """在互联网上搜索实时信息。
    适用于：知识库中没有的外部信息、最新政策法规、行业动态等。
    不适用于：企业内部文档查询（请用 search_knowledge_base）、数学计算、员工数据查询。
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            "web_search 工具未就绪：缺少 duckduckgo-search 依赖。\n"
            "安装命令：pip install duckduckgo-search -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "未找到相关搜索结果。"
        parts = []
        for r in results:
            parts.append(
                f"标题：{r.get('title', '')}\n"
                f"摘要：{r.get('body', '')}\n"
                f"来源：{r.get('href', '')}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"搜索失败：{e}"
