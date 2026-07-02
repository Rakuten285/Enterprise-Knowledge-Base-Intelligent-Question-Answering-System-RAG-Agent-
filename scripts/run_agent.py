"""
Multi-Agent 测试脚本

用法：
    python scripts/run_agent.py "公司年假有多少天"          # 单次问答
    python scripts/run_agent.py --chat                      # 多轮对话模式（验证 Memory）
    python scripts/run_agent.py --demo                      # 跑预设的三类演示问题

同一个 --thread-id 的多次调用共享对话历史（Memory）。
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.core.logging import logger

DEMO_QUESTIONS = [
    # (问题, 说明)
    ("公司年假有多少天？",                     "→ 触发：知识库检索（员工手册）"),
    ("技术部所有员工的平均月薪是多少？",         "→ 触发：数据库查询 + 计算器"),
    ("上一个问题里，技术部工资最高的是谁？",     "→ 触发：Memory 跨轮次引用"),
]


def run_single(query: str, thread_id: str, verbose: bool = True) -> str:
    """执行单次问答，返回最终回答文本"""
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state_input = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "iterations": 0,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"问题：{query}")
        print(f"Thread: {thread_id}")
        print("=" * 60)

    final_answer = ""
    for step in app.stream(state_input, config=config, stream_mode="values"):
        last_msg = step["messages"][-1]
        msg_type = type(last_msg).__name__

        if verbose:
            if msg_type == "AIMessage":
                if getattr(last_msg, "tool_calls", None):
                    tool_name = last_msg.tool_calls[0]["name"]
                    tool_args = last_msg.tool_calls[0]["args"]
                    print(f"\n[Coordinator] 决策：调用 {tool_name}")
                    print(f"  参数：{tool_args}")
                else:
                    print(f"\n[Coordinator] 最终回答生成中...")
            elif msg_type == "ToolMessage":
                preview = last_msg.content[:120].replace("\n", " ")
                print(f"[Tool:{last_msg.name}] 结果：{preview}{'...' if len(last_msg.content) > 120 else ''}")

    # 最后一条 AIMessage 就是最终回答
    for msg in reversed(step["messages"]):
        if type(msg).__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
            break

    if verbose:
        print(f"\n【最终回答】\n{final_answer}")

    return final_answer


def chat_mode(thread_id: str) -> None:
    """多轮对话模式，验证跨轮次 Memory"""
    print(f"\n多轮对话模式（thread_id={thread_id}）")
    print("输入 'exit' 或 'quit' 退出\n")
    while True:
        try:
            query = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        run_single(query, thread_id=thread_id, verbose=True)


def demo_mode() -> None:
    """跑预设的三类演示问题，thread_id 相同以验证 Memory"""
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"\n演示模式（thread_id={thread_id}，三轮共享同一会话）\n")
    for i, (question, hint) in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'#'*60}")
        print(f"演示 {i}/3  {hint}")
        run_single(question, thread_id=thread_id, verbose=True)
        input("\n按 Enter 继续下一个演示...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent 问答系统测试")
    parser.add_argument("query", nargs="?", help="单次提问")
    parser.add_argument("--chat", action="store_true", help="多轮对话模式")
    parser.add_argument("--demo", action="store_true", help="三类演示问题")
    parser.add_argument("--thread-id", default=None, help="会话 ID，相同 ID 共享 Memory")
    args = parser.parse_args()

    thread_id = args.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    if args.demo:
        demo_mode()
    elif args.chat:
        chat_mode(thread_id)
    elif args.query:
        run_single(args.query, thread_id=thread_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
