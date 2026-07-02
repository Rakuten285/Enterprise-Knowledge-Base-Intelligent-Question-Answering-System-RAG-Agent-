"""
全局配置模块
统一从 .env 加载配置，所有模块通过 `from app.core.config import settings` 使用
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- App ----------
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ---------- LLM Provider ----------
    llm_provider: Literal["deepseek", "zhipu", "qwen", "openai"] = Field(
        default="deepseek", alias="LLM_PROVIDER"
    )

    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    zhipu_api_key: str = Field(default="", alias="ZHIPU_API_KEY")
    zhipu_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4", alias="ZHIPU_BASE_URL"
    )
    zhipu_model: str = Field(default="glm-4-flash", alias="ZHIPU_MODEL")

    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # ---------- Embedding Provider ----------
    embedding_provider: Literal["bge_local", "openai", "zhipu"] = Field(
        default="bge_local", alias="EMBEDDING_PROVIDER"
    )
    bge_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5", alias="BGE_MODEL_NAME"
    )
    bge_device: str = Field(default="cpu", alias="BGE_DEVICE")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    zhipu_embedding_model: str = Field(
        default="embedding-3", alias="ZHIPU_EMBEDDING_MODEL"
    )

    # ---------- Chunking / Retrieval ----------
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")

    hybrid_fusion_strategy: Literal["rrf", "weighted"] = Field(
        default="rrf", alias="HYBRID_FUSION_STRATEGY"
    )
    vector_weight: float = Field(default=0.5, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.5, alias="BM25_WEIGHT")
    rrf_k: int = Field(default=60, alias="RRF_K")

    mmr_enabled: bool = Field(default=True, alias="MMR_ENABLED")
    mmr_lambda: float = Field(default=0.7, alias="MMR_LAMBDA")
    mmr_fetch_k: int = Field(default=20, alias="MMR_FETCH_K")

    # ---------- Vector Store ----------
    vectorstore_dir: str = Field(
        default="./data/vectorstore", alias="VECTORSTORE_DIR"
    )
    vectorstore_index_name: str = Field(
        default="kb_index", alias="VECTORSTORE_INDEX_NAME"
    )

    # ---------- Redis ----------
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # ---------- LangSmith ----------
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="enterprise-rag-agent", alias="LANGCHAIN_PROJECT"
    )

    @property
    def vectorstore_path(self) -> Path:
        path = PROJECT_ROOT / self.vectorstore_dir.lstrip("./")
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """缓存单例，避免重复解析 .env"""
    return Settings()


settings = get_settings()
