"""
检索 Agent 节点

职责：执行协调者发出的 search_knowledge_base tool call，
把结果包装成 ToolMessage 追加到状态，然后将控制权交回协调者。
"""
from __future__ import annotations

from langchain_core.messages import ToolMessage

from app.agents.state import AgentState
from app.tools.retrieval_tool import search_knowledge_base


def retrieval_agent_node(state: AgentState) -> dict:
    """从 messages 中取出最新的 tool_call，执行知识库检索，追加 ToolMessage"""
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]

    result = search_knowledge_base.invoke(tool_call["args"])

    tool_message = ToolMessage(
        content=result,
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )
    return {"messages": [tool_message]}
