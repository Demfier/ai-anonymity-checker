"""Factory for creating LLM clients."""

from __future__ import annotations

from ..config import LLMConfig, LLMProvider
from .base import LLMClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration.

    All providers use the same OpenAI-compatible client; the factory
    just ensures sensible defaults are applied.
    """
    return LLMClient(config)
