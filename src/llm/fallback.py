"""
LLM Fallback Chain
===================
Manages multiple LLM providers with automatic fallback.
Tries providers in priority order: Groq → Google → Together.
If all fail, raises an exception.
"""

import asyncio
import traceback

from src.llm.base import BaseLLMClient
from src.llm.groq_client import GroqClient
from src.llm.google_client import GoogleClient
from src.llm.together_client import TogetherClient
from src.config import LLM_PROVIDERS


class LLMFallbackChain:
    """Manages LLM providers with automatic fallback."""

    def __init__(self):
        self._clients: dict[str, BaseLLMClient] = {
            "groq": GroqClient(),
            "google": GoogleClient(),
            "together": TogetherClient(),
        }
        self._last_used: str = ""
        self._error_log: list[dict] = []

    @property
    def available_providers(self) -> list[str]:
        """List providers that have API keys configured."""
        return [
            name for name in LLM_PROVIDERS
            if name in self._clients and self._clients[name].is_available
        ]

    @property
    def last_used(self) -> str:
        """Name of the last successfully used provider."""
        return self._last_used

    def status(self) -> dict:
        """Get status of all providers."""
        return {
            name: {
                "available": client.is_available,
                "name": client.name,
            }
            for name, client in self._clients.items()
        }

    async def generate(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        preferred_provider: str = None,
    ) -> str:
        """
        Generate a response using the fallback chain.

        Tries each provider in order until one succeeds.
        If preferred_provider is specified, try it first.
        """
        # Build provider order
        providers = list(LLM_PROVIDERS)
        if preferred_provider and preferred_provider in providers:
            providers.remove(preferred_provider)
            providers.insert(0, preferred_provider)

        errors = []
        for provider_name in providers:
            client = self._clients.get(provider_name)
            if not client or not client.is_available:
                continue

            try:
                result = await client.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._last_used = provider_name
                return result
            except Exception as e:
                error_info = {
                    "provider": provider_name,
                    "error": str(e),
                    "type": type(e).__name__,
                }
                errors.append(error_info)
                self._error_log.append(error_info)
                print(f"  [Fallback] {client.name} failed: {e}")
                continue

        # All providers failed
        available = self.available_providers
        if not available:
            raise RuntimeError(
                "No LLM providers configured! Add at least one API key to .env:\n"
                "  GROQ_API_KEY=...\n"
                "  GOOGLE_API_KEY=...\n"
                "  TOGETHER_API_KEY=..."
            )

        error_summary = "; ".join(
            f"{e['provider']}: {e['error']}" for e in errors
        )
        raise RuntimeError(f"All LLM providers failed: {error_summary}")

    def generate_sync(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        preferred_provider: str = None,
    ) -> str:
        """Synchronous wrapper around generate()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context; create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate(messages, temperature, max_tokens, preferred_provider),
                )
                return future.result()
        else:
            return asyncio.run(
                self.generate(messages, temperature, max_tokens, preferred_provider)
            )
