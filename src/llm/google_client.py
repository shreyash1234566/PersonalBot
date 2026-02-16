"""
Google AI Studio Client
========================
Google Gemini Flash API client (secondary provider).
Free tier: 15 RPM, 1M tokens/day.
Uses the google-generativeai SDK.
"""

import time

from src.llm.base import BaseLLMClient
from src.config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GOOGLE_MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)


class GoogleClient(BaseLLMClient):
    """Google AI Studio (Gemini) client."""

    def __init__(self):
        self._client = None
        self._last_request_time = 0
        self._min_interval = 4.5  # ~15 RPM → 1 req per 4s (safe margin)

    @property
    def name(self) -> str:
        return f"Google ({GOOGLE_MODEL})"

    @property
    def is_available(self) -> bool:
        return bool(GOOGLE_API_KEY)

    def _get_client(self):
        """Lazy-init the Google AI client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=GOOGLE_API_KEY)
        return self._client

    def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _chatml_to_gemini(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Convert ChatML messages to Gemini's format.
        
        Returns (system_instruction, contents) where contents is a list
        of {"role": "user"|"model", "parts": [{"text": ...}]}.
        """
        system_text = ""
        contents = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_text += text + "\n"
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": text}],
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": text}],
                })

        return system_text.strip(), contents

    async def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Google API key not configured")

        self._rate_limit_wait()
        client = self._get_client()
        system_text, contents = self._chatml_to_gemini(messages)

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
            temperature=temperature or TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=max_tokens or GOOGLE_MAX_TOKENS,
        )

        response = client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=contents,
            config=config,
        )

        return response.text.strip()

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        # google-genai SDK is synchronous by nature, just call generate
        return await self.generate(messages, temperature, max_tokens)
