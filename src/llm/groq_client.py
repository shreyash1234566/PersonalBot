"""
Groq LLM Client
================
Groq API client for Llama 3.3 70B (primary provider).
Free tier: 30 requests/min, 6000 tokens/min.
Uses OpenAI-compatible API format via httpx.
"""

import httpx
import asyncio
import time

from src.llm.base import BaseLLMClient
from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    GROQ_MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
)


class GroqClient(BaseLLMClient):
    """Groq API client (OpenAI-compatible)."""

    def __init__(self):
        self._last_request_time = 0
        self._min_interval = 2.1  # ~30 RPM → 1 req per 2s (safe margin)

    @property
    def name(self) -> str:
        return f"Groq ({GROQ_MODEL})"

    @property
    def is_available(self) -> bool:
        return bool(GROQ_API_KEY)

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or GROQ_MAX_TOKENS,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "stream": False,
        }

    def _rate_limit_wait(self):
        """Respect rate limits by waiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    async def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Groq API key not configured")

        self._rate_limit_wait()
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or GROQ_MAX_TOKENS,
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Groq API key not configured")

        self._rate_limit_wait()
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or GROQ_MAX_TOKENS,
        )

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()
