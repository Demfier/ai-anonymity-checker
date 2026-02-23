"""Configuration management for anonymization checker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"


# Default base URLs for each provider
PROVIDER_BASE_URLS: dict[str, str] = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
    LLMProvider.VLLM: "http://localhost:8000/v1",
}

# Default models for each provider
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.OPENROUTER: "openai/gpt-4o",
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.VLLM: "default",
}


@dataclass
class LLMConfig:
    """LLM backend configuration."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.6
    max_tokens: int = 4096
    timeout: int = 120

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return PROVIDER_BASE_URLS.get(self.provider, "http://localhost:8000/v1")

    def get_model(self) -> str:
        if self.model:
            return self.model
        return PROVIDER_DEFAULT_MODELS.get(self.provider, "default")

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_keys = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            LLMProvider.OLLAMA: "",
            LLMProvider.VLLM: "",
        }
        env_var = env_keys.get(self.provider, "")
        if env_var:
            return os.environ.get(env_var, "")
        return "no-key-needed"


@dataclass
class CheckConfig:
    """Configuration for which checks to run."""

    enabled_checks: list[str] = field(
        default_factory=lambda: [
            "author_names",
            "affiliations",
            "self_citations",
            "acknowledgments",
            "code_repos",
            "contact_info",
            "pdf_metadata",
            "latex_artifacts",
            "deanon_leftovers",
            "funding",
            "arxiv_self_refs",
            "watermarks_headers",
        ]
    )
    confidence_threshold: float = 0.5
    skip_llm_checks: bool = False


@dataclass
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    checks: CheckConfig = field(default_factory=CheckConfig)
    verbose: bool = False
    output_format: str = "both"  # "json", "markdown", "both"
