"""
LangGraph Multi-Agent 图的组装与编译

图结构：
    START → coordinator → (route) → retrieval_agent → coordinator
                                  → executor_agent  → coordinator
                                  → END（直接回答）

Memory：使用 MemorySaver 做状态持久化，同一 thread_id 的多轮对话共享历史。
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.coordinator import coordinator_node, route_after_coordinator
from app.agents.executor_agent import executor_agent_node
from app.agents.retrieval_agent import retrieval_agent_node
from app.agents.state import AgentState


@lru_cache(maxsize=1)
def build_graph():
    """构建并编译 LangGraph 图（单例，避免重复编译）"""
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("executor_agent", executor_agent_node)

    # 入口：从协调者开始
    graph.set_entry_point("coordinator")

    # 协调者 → 条件路由
    graph.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "retrieval_agent": "retrieval_agent",
            "executor_agent": "executor_agent",
            "END": END,
        },
    )

    # 专属 Agent 执行完后，控制权交回协调者
    graph.add_edge("retrieval_agent", "coordinator")
    graph.add_edge("executor_agent", "coordinator")

    # 编译，挂载内存检查点（支持跨轮次 Memory）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
