"""
LangGraph 全局状态定义

messages 使用 operator.add 累加——每个节点追加消息，不覆盖，
这样整条推理链（Human → AI tool_call → Tool result → AI answer）都保留在状态里，
既是 Memory，也是调试时可以完整回放的轨迹。
"""
from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 保存用户原始问题，方便各节点直接取用，不用每次从 messages 里解析
    query: str
    # 防止无限循环的安全阀：协调者每轮 +1，超过阈值强制结束
    iterations: int
