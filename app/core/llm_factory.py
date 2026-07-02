"""
LLM Provider 工厂
DeepSeek / 智谱GLM / 通义千问 均提供 OpenAI 兼容的 /v1/chat/completions 接口，
因此统一通过 langchain_openai.ChatOpenAI 调用，仅切换 base_url / api_key / model。
默认使用 DeepSeek（deepseek-chat），性价比高且有免费额度。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import settings
from app.core.logging import logger

_PROVIDER_CONFIG = {
    "deepseek": lambda: (
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        settings.deepseek_model,
    ),
    "zhipu": lambda: (
        settings.zhipu_api_key,
        settings.zhipu_base_url,
        settings.zhipu_model,
    ),
    "qwen": lambda: (
        settings.qwen_api_key,
        settings.qwen_base_url,
        settings.qwen_model,
    ),
    "openai": lambda: (
        settings.openai_api_key,
        settings.openai_base_url,
        settings.openai_model,
    ),
}


@lru_cache
def get_llm(
    provider: str | None = None,
    temperature: float = 0.1,
    streaming: bool = False,
) -> BaseChatModel:
    """
    获取 LLM 实例。

    Args:
        provider: deepseek / zhipu / qwen / openai，默认读取 settings.llm_provider
        temperature: 采样温度，RAG 问答场景建议调低以减少幻觉
        streaming: 是否启用流式输出（配合 FastAPI SSE 接口使用）
    """
    from langchain_openai import ChatOpenAI

    selected = provider or settings.llm_provider
    if selected not in _PROVIDER_CONFIG:
        raise ValueError(f"未知的 LLM_PROVIDER: {selected}，支持: {list(_PROVIDER_CONFIG)}")

    api_key, base_url, model = _PROVIDER_CONFIG[selected]()
    if not api_key:
        raise ValueError(
            f"LLM_PROVIDER={selected} 但未配置对应的 API_KEY，请检查 .env 文件"
        )

    logger.info(f"初始化 LLM: provider={selected}, model={model}, streaming={streaming}")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        streaming=streaming,
    )
