"""
/chat 接口：SSE 流式问答

两种模式：
  mode=agent  → 走 LangGraph Multi-Agent 图（默认）
  mode=rag    → 直接走 RAG 检索+生成，不经过 Agent
"""
from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.api.models import ChatRequest, ChatResponse
from app.core.logging import logger
from app.rag.chain import generate_answer
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])

# 进程级单例，避免每次请求重建索引
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.load()
    return _pipeline


# --------------------------------------------------------------------------
# SSE 生成器：Agent 模式
# --------------------------------------------------------------------------
async def _agent_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "iterations": 0,
    }

    final_answer = ""
    try:
        for step in app.stream(state_input, config=config, stream_mode="values"):
            last = step["messages"][-1]
            msg_type = type(last).__name__

            if msg_type == "AIMessage":
                if getattr(last, "tool_calls", None):
                    tool_name = last.tool_calls[0]["name"]
                    args = last.tool_calls[0]["args"]
                    event = {"event": "tool_call", "data": {"tool": tool_name, "args": args}}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                else:
                    final_answer = last.content
            elif msg_type == "ToolMessage":
                event = {"event": "tool_result", "data": {"tool": last.name, "preview": last.content[:200]}}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 按 token 逐字流式输出最终回答
        for char in final_answer:
            event = {"event": "token", "data": char}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        done = {"event": "done", "data": {"thread_id": thread_id, "answer": final_answer}}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Agent 流式处理出错: {e}")
        err = {"event": "error", "data": str(e)}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------
# SSE 生成器：RAG 模式
# --------------------------------------------------------------------------
async def _rag_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    try:
        pipeline = _get_pipeline()
        from app.core.config import settings
        chunks = pipeline.retrieve(query, k=settings.top_k)
        rag_answer = generate_answer(query, chunks)

        for char in rag_answer.answer:
            event = {"event": "token", "data": char}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        done = {
            "event": "done",
            "data": {
                "thread_id": thread_id,
                "answer": rag_answer.answer,
                "sources": rag_answer.sources,
            },
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"RAG 流式处理出错: {e}")
        err = {"event": "error", "data": str(e)}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@router.post("", summary="流式问答（SSE）")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or f"session-{uuid.uuid4().hex[:8]}"
    logger.info(f"[{req.mode}] query={req.query[:50]} thread={thread_id}")

    if req.mode == "rag":
        generator = _rag_stream(req.query, thread_id)
    else:
        generator = _agent_stream(req.query, thread_id)

    if req.stream:
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式：消费完 generator，返回完整 JSON
    full_answer = ""
    sources: list[str] = []
    async for chunk in generator:
        if not chunk.startswith("data:"):
            continue
        payload = json.loads(chunk[len("data: "):].strip())
        if payload["event"] == "done":
            full_answer = payload["data"].get("answer", "")
            sources = payload["data"].get("sources", [])

    return ChatResponse(
        answer=full_answer,
        thread_id=thread_id,
        sources=sources,
        mode=req.mode,
    )
