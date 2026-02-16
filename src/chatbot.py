"""
Chatbot Orchestrator
====================
Main chatbot class that ties everything together:
  VectorStore → ContextBuilder → LLM → PostProcessor → History

This is the single entry point for generating responses.
Supports both interactive CLI mode and programmatic API.
"""

import asyncio
import traceback  # <--- ADDED for detailed error logging
from datetime import datetime
from pathlib import Path

from src.config import RETRIEVAL_TOP_K, HISTORY_WINDOW
from src.memory.vector_store import VectorStore
from src.memory.history import ConversationHistory
from src.integrations.sheets_logger import SheetsLogger
from src.llm.fallback import LLMFallbackChain
from src.engine.context_builder import ContextBuilder
from src.engine.post_processor import PostProcessor
from typing import Optional


class Chatbot:
    """Shreyash's digital twin chatbot."""

    def __init__(self, history=None, vector_store=None, sheets_logger=None):
        self.vector_store = vector_store or VectorStore()
        self.history = history or ConversationHistory()
        self.sheets_logger = sheets_logger or SheetsLogger()
        self.llm = LLMFallbackChain()
        self.context_builder = ContextBuilder()
        self.post_processor = PostProcessor()

        self._conversation_id: str = "default"
        self._partner_name: str = "a girl"

    # ── Setup ─────────────────────────────────────────────────────────────

    def set_conversation(
        self,
        conversation_id: str,
        partner_name: str = "a girl",
    ):
        """Set the active conversation context."""
        self._conversation_id = conversation_id
        self._partner_name = partner_name
        
        # LOGGING ADDED: Check if DB connection works during setup
        print(f"[Chatbot] Setting conversation to: {conversation_id} ({partner_name})")
        try:
            self.history.get_or_create_conversation(conversation_id, partner_name)
        except Exception as e:
            print(f"[Chatbot] ❌ CRITICAL: Failed to init conversation in DB: {e}")
            traceback.print_exc()

    def status(self) -> dict:
        """Get chatbot system status."""
        stats = {}
        try:
            stats = self.history.get_stats(self._conversation_id)
        except Exception as e:
            stats = {"error": str(e)}

        return {
            "llm_providers": self.llm.status(),
            "available_providers": self.llm.available_providers,
            "vector_store": self.vector_store.info(),
            "conversation_id": self._conversation_id,
            "partner_name": self._partner_name,
            "history_stats": stats,
        }

    # ── Core Response Generation ──────────────────────────────────────────

    

    async def respond(
        self,
        girl_message: str,
        conversation_id: Optional[str] = None,
        partner_name: Optional[str] = None,
    ) -> list[str]:
        """
        Generate Shreyash's response to a girl's message.
        """
        conv_id = conversation_id or self._conversation_id
        partner = partner_name or self._partner_name

        print(f"[Chatbot] Processing message from {conv_id}: '{girl_message[:20]}...'")

        # 0. Ensure conversation exists
        try:
            self.history.get_or_create_conversation(conv_id, partner)
        except Exception as e:
            print(f"[Chatbot] ❌ ERROR: Could not create/get conversation: {e}")

        # 1. Store girl's message (CRITICAL WRITE 1)
        print("[Chatbot] Saving USER message to history...")
        try:
            self.history.add_message(
                conversation_id=conv_id,
                role="user",
                content=girl_message,
            )
            print("[Chatbot] ✅ USER message saved.")
        except Exception as e:
            print(f"[Chatbot] ❌ FAILED to save USER message to DB: {e}")
            traceback.print_exc()

        # 1.5 Optional Sheet Logging
        if self.sheets_logger.enabled:
            try:
                self.sheets_logger.append_message(
                    conversation_id=conv_id,
                    partner_name=partner,
                    role="user",
                    content=girl_message,
                )
            except Exception as e:
                print(f"[Chatbot] Sheet logging failed: {e}")

        # 2. Retrieve similar examples
        retrieved = []
        try:
            if self.vector_store.count() > 0:
                retrieved = self.vector_store.retrieve(
                    query=girl_message,
                    top_k=RETRIEVAL_TOP_K,
                )
        except Exception as e:
            print(f"[Chatbot] Vector retrieval failed (non-fatal): {e}")

        # 3. Get conversation history
        history_turns = []
        try:
            history_turns = self.history.get_recent_as_chatml(
                conversation_id=conv_id,
                limit=HISTORY_WINDOW,
            )
            # Remove the last turn (it's the girl_message we just added)
            if history_turns and history_turns[-1]["content"] == girl_message:
                history_turns = history_turns[:-1]
        except Exception as e:
             print(f"[Chatbot] ❌ Failed to fetch history context: {e}")

        # 4. Build prompt
        messages = self.context_builder.build_messages(
            girl_message=girl_message,
            partner_name=partner,
            history=history_turns,
            retrieved_examples=retrieved,
        )

        # 5. Generate via LLM
        print("[Chatbot] Generating response via LLM...")
        raw_output = await self.llm.generate(messages)

        # 6. Post-process (pass girl_message for hmm↔mm mirroring)
        processed = self.post_processor.process(raw_output, girl_message)

        # 7. Store response in history (CRITICAL WRITE 2)
        full_response = " [MSG_BREAK] ".join(processed)
        
        print(f"[Chatbot] Saving BOT response: '{full_response[:20]}...'")
        try:
            self.history.add_message(
                conversation_id=conv_id,
                role="assistant",
                content=full_response,
                metadata={
                    "provider": self.llm.last_used,
                    "raw_output": raw_output[:500],
                    "retrieved_count": len(retrieved),
                },
            )
            print("[Chatbot] ✅ BOT response saved.")
        except Exception as e:
            print(f"[Chatbot] ❌ FAILED to save BOT message to DB: {e}")
            traceback.print_exc()

        if self.sheets_logger.enabled:
            try:
                self.sheets_logger.append_message(
                    conversation_id=conv_id,
                    partner_name=partner,
                    role="assistant",
                    content=full_response,
                    provider=self.llm.last_used or "",
                )
            except Exception as e:
                print(f"[Chatbot] Sheet logging failed: {e}")

        return processed

    def respond_sync(
        self,
        girl_message: str,
        conversation_id: Optional[str] = None,
        partner_name: Optional[str] = None,
    ) -> list[str]:
        """Synchronous wrapper around respond()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.respond(girl_message, conversation_id, partner_name),
                )
                return future.result()
        else:
            return asyncio.run(
                self.respond(girl_message, conversation_id, partner_name)
            )

    # ── Interactive CLI ───────────────────────────────────────────────────

    def chat_cli(self):
        """Run interactive chat in the terminal."""
        print()
        print("█" * 55)
        print("  SHREYASH BOT — Digital Twin")
        print("█" * 55)
        print()

        # Show status
        status = self.status()
        providers = status["available_providers"]
        if not providers:
            print("  ⚠  No LLM providers configured!")
            print("  Add API keys to .env file:")
            print("    GROQ_API_KEY=...")
            print("    GOOGLE_API_KEY=...")
            print("    TOGETHER_API_KEY=...")
            print()
            return

        print(f"  LLM providers: {', '.join(providers)}")
        print(f"  Example bank: {status['vector_store']['count']:,} examples indexed")
        print(f"  Conversation: {self._conversation_id}")
        print()
        print("  Type a message as the girl. Shreyash will respond.")
        print("  Commands: /new, /status, /history, /clear, /quit")
        print("─" * 55)
        print()

        while True:
            try:
                girl_input = input("  Her: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Bye! 👋")
                break

            if not girl_input:
                continue

            # Handle commands
            if girl_input.startswith("/"):
                self._handle_command(girl_input)
                continue

            # Generate response
            try:
                responses = self.respond_sync(girl_input)
                print()
                for msg in responses:
                    print(f"  Shreyash: {msg}")
                print()
                print(f"  [{self.llm.last_used}]")
                print()
            except Exception as e:
                print(f"\n  ❌ Error: {e}\n")

    def _handle_command(self, cmd: str):
        """Handle CLI commands."""
        cmd = cmd.lower().strip()

        if cmd == "/quit" or cmd == "/exit":
            print("\n  Bye! 👋")
            raise SystemExit

        elif cmd == "/status":
            status = self.status()
            print(f"\n  Providers: {status['available_providers']}")
            print(f"  Last used: {self.llm.last_used or 'none'}")
            print(f"  Examples indexed: {status['vector_store']['count']:,}")
            print(f"  Conversation: {self._conversation_id}")
            stats = status.get("history_stats", {})
            print(f"  Messages in history: {stats.get('message_count', 0)}")
            print()

        elif cmd == "/history":
            msgs = self.history.get_recent_messages(self._conversation_id, limit=20)
            if not msgs:
                print("\n  No history yet.\n")
            else:
                print()
                for m in msgs:
                    role = "Her" if m["role"] == "user" else "Shreyash"
                    print(f"  {role}: {m['content'][:80]}")
                print()

        elif cmd == "/clear":
            self.history.clear_conversation(self._conversation_id)
            print("\n  History cleared.\n")

        elif cmd.startswith("/new"):
            parts = cmd.split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else "girl_" + datetime.now().strftime("%H%M")
            self.set_conversation(name, name)
            print(f"\n  New conversation: {name}\n")

        else:
            print("\n  Commands: /new [name], /status, /history, /clear, /quit\n")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        """Clean up resources."""
        self.history.close()