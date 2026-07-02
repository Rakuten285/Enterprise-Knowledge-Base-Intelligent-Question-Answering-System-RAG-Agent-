"""
协调者节点（Coordinator Agent）

职责：
- 接收用户问题和当前对话历史
- 决定下一步：调用哪个工具，还是已有足够信息可以直接回答
- 通过 LLM 的 tool_calling 能力输出结构化决策，LangGraph 读取 tool_calls 做路由

工作模式（ReAct 风格）：
    用户问题 → Coordinator 决策 → 专属 Agent 执行工具 → 结果返回 Coordinator → 再次决策 → ... → 最终回答
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage

from app.agents.state import AgentState
from app.core.llm_factory import get_llm
from app.tools.calculator_tool import calculate
from app.tools.database_tool import query_employee_database
from app.tools.retrieval_tool import search_knowledge_base
from app.tools.search_tool import web_search

MAX_ITERATIONS = 6  # 最多迭代 6 次防止死循环

_SYSTEM_PROMPT = """你是一个企业知识库智能助手的协调者（Coordinator Agent）。
你需要分析用户问题，选择合适的工具获取信息，最终给出完整准确的回答。

可用工具：
- search_knowledge_base：查询企业内部文档（员工手册、IT安全规范、采购管理制度）
- calculate：计算数学表达式（加减乘除、乘方、取模）
- query_employee_database：查询员工数据库（姓名、部门、职位、月薪、入职日期）
- web_search：搜索互联网实时信息

决策原则：
1. 优先用内部知识库（search_knowledge_base）回答内部制度类问题
2. 涉及具体员工数据时用 query_employee_database
3. 需要计算时用 calculate
4. 只有内部工具都无法满足时才用 web_search
5. 已获得足够信息时，直接回答，不要重复调用工具
6. 同一个工具对同一个问题最多调用 2 次"""

_TOOLS = [search_knowledge_base, calculate, query_employee_database, web_search]


def coordinator_node(state: AgentState) -> dict:
    """协调者节点：调用 LLM 决策下一步行动"""
    iterations = state.get("iterations", 0)

    # 超过最大迭代次数，强制用现有信息作答
    if iterations >= MAX_ITERATIONS:
        from langchain_core.messages import HumanMessage
        force_msg = HumanMessage(
            content="请根据已收集到的信息，给出最终回答。不要再调用工具。"
        )
        messages = list(state["messages"]) + [force_msg]
        llm = get_llm()
        response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT)] + messages)
        return {"messages": [response], "iterations": iterations + 1}

    llm_with_tools = get_llm().bind_tools(_TOOLS)
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iterations": iterations + 1}


def route_after_coordinator(state: AgentState) -> str:
    """条件路由：根据协调者的 tool_calls 决定下一个节点"""
    last = state["messages"][-1]

    # 没有 tool_calls 说明协调者直接给出了最终回答
    if not getattr(last, "tool_calls", None):
        return "END"

    tool_name = last.tool_calls[0]["name"]
    if tool_name == "search_knowledge_base":
        return "retrieval_agent"
    if tool_name in ("calculate", "query_employee_database", "web_search"):
        return "executor_agent"
    return "END"
