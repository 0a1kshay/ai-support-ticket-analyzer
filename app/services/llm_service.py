"""
LLM provider abstraction layer.

Architecture:
    LLMProvider (ABC)
    ├── GroqProvider     – default; uses groq free-tier (llama3-8b-8192)
    ├── OllamaProvider   – local Ollama instance
    └── OpenAIProvider   – OpenAI-compatible endpoint

The application selects the provider based on environment variables.
Never allow the LLM to execute code or perform calculations.
"""
from __future__ import annotations
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Send system + user messages and return the raw text response.
        Raises LLMUnavailableError if the provider is unreachable.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider can accept requests right now."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""


class LLMUnavailableError(Exception):
    """Raised when the LLM backend cannot be reached."""


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """
    Uses the Groq Cloud API (free tier).
    Model: llama3-8b-8192 – fast inference, good JSON adherence.
    Set GROQ_API_KEY in .env to enable.
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self._client = None
        if self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                logger.warning("groq package not installed.")
            except Exception as exc:
                logger.warning("Could not initialise Groq client: %s", exc)

    @property
    def name(self) -> str:
        return f"Groq ({self._model})"

    def is_available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def complete(self, system_prompt: str, user_message: str) -> str:
        if not self.is_available():
            raise LLMUnavailableError(
                "Groq provider is not configured. Set GROQ_API_KEY in .env."
            )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise LLMUnavailableError(f"Groq API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """
    Uses a locally running Ollama instance.
    Start Ollama and pull a model: ollama pull llama3.2
    Set OLLAMA_BASE_URL and OLLAMA_MODEL in .env.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        self._model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)

    @property
    def name(self) -> str:
        return f"Ollama ({self._model})"

    def is_available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            import httpx
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
            r = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            raise LLMUnavailableError(f"Ollama error: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    Generic OpenAI-compatible provider.
    Works with OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint.
    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL, OPENAI_MODEL.
    """

    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._client = None
        if self._api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)
            except Exception as exc:
                logger.warning("Could not init OpenAI client: %s", exc)

    @property
    def name(self) -> str:
        return f"OpenAI ({self._model})"

    def is_available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def complete(self, system_prompt: str, user_message: str) -> str:
        if not self.is_available():
            raise LLMUnavailableError("OpenAI provider not configured.")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI error: %s", exc)
            raise LLMUnavailableError(f"OpenAI error: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    """
    Select the LLM provider based on environment configuration.
    Priority: GROQ → OPENAI → OLLAMA
    """
    provider_name = os.getenv("LLM_PROVIDER", "").lower()

    if provider_name == "ollama":
        return OllamaProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "groq":
        return GroqProvider()

    # Auto-detect: try Groq first (free tier), then OpenAI, then Ollama
    groq = GroqProvider()
    if groq.is_available():
        logger.info("Auto-selected LLM provider: Groq")
        return groq

    openai = OpenAIProvider()
    if openai.is_available():
        logger.info("Auto-selected LLM provider: OpenAI")
        return openai

    ollama = OllamaProvider()
    logger.info(
        "Auto-selected LLM provider: Ollama (available=%s)", ollama.is_available()
    )
    return ollama


# ---------------------------------------------------------------------------
# Intent parsing service
# ---------------------------------------------------------------------------

class IntentParser:
    """
    Converts a natural-language question into a StructuredQuery via the LLM.

    The LLM only performs intent/filter extraction.
    All numerical computation is done deterministically by the QueryEngine.
    """

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def parse(self, question: str) -> dict:
        """
        Returns a raw dict that will be validated by Pydantic StructuredQuery.
        Raises LLMUnavailableError if the provider is unavailable.
        Raises ValueError if the response cannot be parsed as JSON.
        """
        from app.prompts.query_prompt import SYSTEM_PROMPT

        raw = self._provider.complete(SYSTEM_PROMPT, question)
        logger.debug("LLM raw response: %s", raw[:500])

        # Strip markdown fences if model wraps output
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON: %s", raw[:300])
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
