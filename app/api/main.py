"""
FastAPI 应用入口

启动：
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

接口文档：
    http://localhost:8000/docs
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, kb
from app.core.config import settings
from app.core.logging import logger

# --------------------------------------------------------------------------
# LangSmith 链路追踪（在 import LangChain 模块前设置环境变量）
# --------------------------------------------------------------------------
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info(f"LangSmith 追踪已启用，项目：{settings.langchain_project}")

# --------------------------------------------------------------------------
# FastAPI 实例
# --------------------------------------------------------------------------
app = FastAPI(
    title="企业知识库智能问答系统",
    description="RAG + Multi-Agent 企业内部知识库问答，支持流式输出",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# 请求耗时日志
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
    return response

# --------------------------------------------------------------------------
# 路由注册
# --------------------------------------------------------------------------
app.include_router(chat.router, prefix="/api/v1")
app.include_router(kb.router, prefix="/api/v1")

# --------------------------------------------------------------------------
# 基础端点
# --------------------------------------------------------------------------
@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "enterprise-rag-agent", "version": "1.0.0"}

@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy", "llm_provider": settings.llm_provider}

# --------------------------------------------------------------------------
# 全局异常处理
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常 {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
