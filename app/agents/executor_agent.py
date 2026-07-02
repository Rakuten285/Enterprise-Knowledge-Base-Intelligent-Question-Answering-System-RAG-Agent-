"""
执行 Agent 节点

职责：执行计算器、数据库查询、网络搜索三类工具。
从最新的 tool_call 中读取工具名称，分发到对应工具，追加 ToolMessage。
"""
from __future__ import annotations

from langchain_core.messages import ToolMessage

from app.agents.state import AgentState
from app.tools.calculator_tool import calculate
from app.tools.database_tool import query_employee_database
from app.tools.search_tool import web_search

_TOOL_MAP = {
    "calculate": calculate,
    "query_employee_database": query_employee_database,
    "web_search": web_search,
}


def executor_agent_node(state: AgentState) -> dict:
    """根据 tool_call 名称分发到对应工具，追加执行结果为 ToolMessage"""
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]
    tool_name = tool_call["name"]

    tool_fn = _TOOL_MAP.get(tool_name)
    if tool_fn is None:
        result = f"未知工具：{tool_name}"
    else:
        result = tool_fn.invoke(tool_call["args"])

    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
        name=tool_name,
    )
    return {"messages": [tool_message]}
