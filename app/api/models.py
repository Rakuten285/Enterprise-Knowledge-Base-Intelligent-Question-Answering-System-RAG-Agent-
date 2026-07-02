"""API 请求/响应数据模型"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(None, description="会话ID，相同ID共享Memory；不传则自动生成")
    mode: str = Field("agent", description="问答模式：agent（多智能体）或 rag（纯检索生成）")
    stream: bool = Field(True, description="是否流式输出")


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str] = []
    mode: str = "agent"


class SSEEvent(BaseModel):
    event: str  # "token" | "done" | "error" | "tool_call"
    data: Any


class KBStatus(BaseModel):
    total_chunks: int
    documents: list[str]
    vectorstore_exists: bool
    bm25_exists: bool


class BuildRequest(BaseModel):
    chunk_size: int = Field(800, ge=100, le=2000)
    chunk_overlap: int = Field(50, ge=0, le=200)


class BuildResponse(BaseModel):
    success: bool
    total_chunks: int
    message: str
