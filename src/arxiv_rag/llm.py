"""Ollama HTTP client (local, Open-source models)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from arxiv_rag.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        ...

    def is_available(self) -> bool:
        ...


@dataclass
class OllamaError(RuntimeError):
    message: str
    status_code: int | None = None

    def __str__(self) -> str:  # pragma: no cover
        return self.message


class OllamaClient:
    """Talks to a local Ollama server over HTTP (default localhost:11434)."""

    def __init__(self, cfg: LLMConfig):
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.timeout_seconds

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            return [m.get("name", "") for m in payload.get("models", [])]
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot list Ollama models: {exc}") from exc

    def generate(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise OllamaError(
                "Cannot reach Ollama at "
                f"{self.base_url}. Install Ollama, start the server "
                "(`ollama serve`), and pull the model "
                f"(`ollama pull {self.model}`)."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:400]}",
                status_code=response.status_code,
            )
        payload = response.json()
        message = payload.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise OllamaError("Ollama returned an empty response.")
        return content


class StaticLLMClient:
    """Deterministic stand-in used in unit tests."""

    def __init__(self, reply: str = "This is a mocked answer [0000.00001]."):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def is_available(self) -> bool:
        return True

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply
