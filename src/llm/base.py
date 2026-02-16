"""
Base LLM Client
================
Abstract interface for LLM providers. All providers implement this.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured with a valid API key."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.75,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: ChatML-format messages [{role, content}, ...]
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            Generated text string

        Raises:
            Exception on API errors (caller handles fallback)
        """
        ...

    @abstractmethod
    async def generate_sync(
        self,
        messages: list[dict],
        temperature: float = 0.75,
        max_tokens: int = 512,
    ) -> str:
        """Synchronous version of generate."""
        ...
