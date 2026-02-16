"""
Together AI Client
===================
Together AI API client for Qwen2.5-72B (tertiary fallback).
Free tier: $25 credit on signup, competitive pricing after.
Uses OpenAI-compatible API format.
"""

import httpx
import time

from src.llm.base import BaseLLMClient
from src.config import (
    TOGETHER_API_KEY,
    TOGETHER_MODEL,
    TOGETHER_BASE_URL,
    TOGETHER_MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
)


class TogetherClient(BaseLLMClient):
    """Together AI client (OpenAI-compatible)."""

    def __init__(self):
        self._last_request_time = 0
        self._min_interval = 1.1  # ~60 RPM

    @property
    def name(self) -> str:
        return f"Together ({TOGETHER_MODEL})"

    @property
    def is_available(self) -> bool:
        return bool(TOGETHER_API_KEY)

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {
            "model": TOGETHER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or TOGETHER_MAX_TOKENS,
            "top_p": TOP_P,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "stream": False,
        }

    def _rate_limit_wait(self):
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
            raise RuntimeError("Together API key not configured")

        self._rate_limit_wait()
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or TOGETHER_MAX_TOKENS,
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{TOGETHER_BASE_URL}/chat/completions",
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
            raise RuntimeError("Together API key not configured")

        self._rate_limit_wait()
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or TOGETHER_MAX_TOKENS,
        )

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{TOGETHER_BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()
