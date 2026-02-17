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

# Use monotonic clock for rate limiting (immune to system clock changes)
_clock = time.monotonic
from src.config import (
    GROQ_API_KEYS,
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
    """Groq API client (OpenAI-compatible) with automatic multi-key rotation.
    
    Supports unlimited API keys. Add keys in .env using any format:
      GROQ_API_KEYS=key1,key2,key3       (comma-separated — easiest)
      GROQ_API_KEY_1=..., GROQ_API_KEY_2=... (numbered — unlimited)
      GROQ_API_KEY=...                    (single — backward-compatible)
    
    On 429 (rate limit), automatically rotates to the next key.
    """

    def __init__(self):
        self._last_request_time = 0
        self._min_interval = 2.1  # ~30 RPM → 1 req per 2s (safe margin)
        
        # Multi-key rotation support
        self.keys = list(GROQ_API_KEYS) if GROQ_API_KEYS else ([GROQ_API_KEY] if GROQ_API_KEY else [])
        self.current_key_index = 0
        self._total_requests = 0
        self._total_rotations = 0
        self.key_stats = {i: {"ok": 0, "429": 0} for i in range(len(self.keys))}
        
        if len(self.keys) > 1:
            print(f"  [Groq] Multi-key rotation enabled: {len(self.keys)} keys loaded")

    @property
    def name(self) -> str:
        return f"Groq ({GROQ_MODEL})"

    @property
    def is_available(self) -> bool:
        return len(self.keys) > 0

    def _get_headers(self) -> dict:
        """Get headers with current API key for rotation."""
        current_key = self.keys[self.current_key_index] if self.keys else ""
        return {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
        }
    
    def _rotate_key(self):
        """Switch to next available API key. Returns True if wrapped around."""
        if len(self.keys) <= 1:
            return False
        old_idx = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        self._total_rotations += 1
        wrapped = self.current_key_index <= old_idx  # True if cycled back to start
        print(f"  [Groq] Rotated to key #{self.current_key_index + 1}/{len(self.keys)} "
              f"(429s on this key: {self.key_stats[self.current_key_index]['429']})")
        return wrapped
    
    def get_stats(self) -> dict:
        """Return key rotation statistics for monitoring."""
        return {
            "total_keys": len(self.keys),
            "current_key": self.current_key_index + 1,
            "total_requests": self._total_requests,
            "total_rotations": self._total_rotations,
            "per_key": {
                f"key_{i+1}": self.key_stats[i]
                for i in range(len(self.keys))
            },
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

    async def _rate_limit_wait_async(self):
        """Non-blocking rate-limit wait (server-friendly)."""
        now = _clock()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = _clock()

    def _rate_limit_wait_sync(self):
        """Blocking rate-limit wait (for sync callers)."""
        now = _clock()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = _clock()

    async def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Groq API keys not configured")

        await self._rate_limit_wait_async()
        self._total_requests += 1
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or GROQ_MAX_TOKENS,
        )

        # Try each key once; on 429 rotate to next key automatically
        max_attempts = max(len(self.keys), 1)
        
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(max_attempts):
                resp = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                
                if resp.status_code == 429:
                    self.key_stats[self.current_key_index]["429"] += 1
                    if len(self.keys) > 1 and attempt < max_attempts - 1:
                        self._rotate_key()
                        continue
                
                resp.raise_for_status()
                self.key_stats[self.current_key_index]["ok"] += 1
                return resp.json()["choices"][0]["message"]["content"].strip()
        
        # All keys exhausted — raise the last 429
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError("Groq API keys not configured")

        self._rate_limit_wait_sync()
        self._total_requests += 1
        payload = self._build_payload(
            messages,
            temperature or TEMPERATURE,
            max_tokens or GROQ_MAX_TOKENS,
        )

        # Try each key once; on 429 rotate to next key automatically
        max_attempts = max(len(self.keys), 1)
        
        with httpx.Client(timeout=30) as client:
            for attempt in range(max_attempts):
                resp = client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                
                if resp.status_code == 429:
                    self.key_stats[self.current_key_index]["429"] += 1
                    if len(self.keys) > 1 and attempt < max_attempts - 1:
                        self._rotate_key()
                        continue
                
                resp.raise_for_status()
                self.key_stats[self.current_key_index]["ok"] += 1
                return resp.json()["choices"][0]["message"]["content"].strip()
        
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
