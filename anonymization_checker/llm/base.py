"""OpenAI-compatible LLM client.

All providers (OpenAI, OpenRouter, Ollama, vLLM) use the same OpenAI-compatible
chat completions API, just with different base URLs.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any

from ..config import LLMConfig

logger = logging.getLogger(__name__)

_has_openai = False
try:
    from openai import OpenAI

    _has_openai = True
except ImportError:
    pass


class LLMClient:
    """Unified LLM client using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.get_model()
        self._api_key = config.get_api_key() or "no-key-needed"
        self._base_url = config.get_base_url().rstrip("/")

        if _has_openai:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=config.timeout,
            )
        else:
            self._client = None

    def complete(
        self,
        prompt: str | list[dict[str, Any]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        plugins = None
        if "openrouter.ai" in self._base_url:
            if isinstance(prompt, list):
                if any(
                    isinstance(item, dict) and item.get("type") == "file"
                    for item in prompt
                ):
                    plugins = [{"id": "file-parser", "pdf": {"engine": "native"}}]

        if self._client is not None:
            return self._complete_openai(
                messages, temp, tokens, response_format, plugins
            )
        else:
            return self._complete_urllib(
                messages, temp, tokens, response_format, plugins
            )

    def _complete_openai(
        self, messages, temperature, max_tokens, response_format, plugins=None
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if plugins:
            kwargs["extra_body"] = {"plugins": plugins}
        try:
            response = self._client.chat.completions.create(**kwargs)  # type: ignore
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def _complete_urllib(
        self, messages, temperature, max_tokens, response_format, plugins=None
    ):
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if plugins:
            payload["plugins"] = plugins

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return (body["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def complete_json(
        self,
        prompt: str | list[dict[str, Any]],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict | list:
        """Send a completion and parse the response as JSON."""
        # First try: ask for JSON format
        try:
            text = self.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return json.loads(text)
        except json.JSONDecodeError:
            # Got a response but it wasn't valid JSON — try to extract it
            return self._extract_json(text)
        except Exception:
            pass

        # Second try: no response_format (some providers don't support it)
        text = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract JSON from text that may contain markdown code blocks."""
        for marker in ("```json", "```"):
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        for i, ch in enumerate(text):
            if ch in ("{", "["):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    pass

        return {"violations": [], "error": "Failed to parse LLM JSON response"}
