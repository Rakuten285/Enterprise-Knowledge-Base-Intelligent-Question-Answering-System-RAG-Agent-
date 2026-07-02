"""
/kb 接口：知识库管理

  GET  /kb/status   查看知识库状态（分块数、文档列表）
  POST /kb/build    重建知识库索引（从 data/raw/ 读取文档）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.models import BuildRequest, BuildResponse, KBStatus
from app.core.config import settings
from app.core.logging import logger
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"


@router.get("/status", response_model=KBStatus, summary="查看知识库状态")
async def kb_status():
    vs_path = settings.vectorstore_path
    # LangChain FAISS.save_local 产生的是子目录结构：kb_index/index.faiss + kb_index/index.pkl
    index_file = vs_path / settings.vectorstore_index_name / "index.faiss"
    bm25_file = vs_path / f"{settings.vectorstore_index_name}_bm25.pkl"

    docs = [f.name for f in RAW_DIR.iterdir() if f.suffix in (".md", ".pdf", ".docx", ".txt")] if RAW_DIR.exists() else []

    # 估算分块数：尝试加载 FAISS 获取真实数量
    total_chunks = 0
    if index_file.exists():
        try:
            import faiss
            idx = faiss.read_index(str(index_file))
            total_chunks = idx.ntotal
        except Exception:
            total_chunks = -1  # 无法读取时返回 -1

    return KBStatus(
        total_chunks=total_chunks,
        documents=sorted(docs),
        vectorstore_exists=index_file.exists(),
        bm25_exists=bm25_file.exists(),
    )


@router.post("/build", response_model=BuildResponse, summary="重建知识库索引")
async def kb_build(req: BuildRequest):
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        raise HTTPException(status_code=400, detail=f"data/raw/ 目录为空，请先上传文档")

    logger.info(f"开始重建知识库: chunk_size={req.chunk_size}, overlap={req.chunk_overlap}")
    try:
        pipeline = RAGPipeline()
        n = pipeline.build_from_directory(RAW_DIR, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)

        # 更新进程级单例
        from app.api.routes.chat import _pipeline
        import app.api.routes.chat as chat_module
        chat_module._pipeline = pipeline

        return BuildResponse(success=True, total_chunks=n, message=f"知识库重建完成，共 {n} 个分块")
    except Exception as e:
        logger.error(f"知识库重建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
