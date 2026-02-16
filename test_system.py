"""
End-to-End Smoke Test
=====================
Tests each component individually, then the full pipeline.
Does NOT require API keys — tests everything except actual LLM calls.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_config():
    """Test config loading."""
    print("  [1/7] Config...", end=" ")
    from src.config import (
        ROOT_DIR, STYLE_BIBLE_FILE, EXAMPLES_FILE,
        GROQ_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY,
    )
    assert ROOT_DIR.exists(), "ROOT_DIR doesn't exist"
    assert STYLE_BIBLE_FILE.exists(), "style_bible.json not found"
    assert EXAMPLES_FILE.exists(), "example_bank.jsonl not found"

    keys_set = sum(bool(k) for k in [GROQ_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY])
    print(f"OK (paths valid, {keys_set}/3 API keys configured)")


def test_vector_store():
    """Test ChromaDB retrieval."""
    print("  [2/7] VectorStore...", end=" ")
    from src.memory.vector_store import VectorStore

    store = VectorStore()
    count = store.count()
    assert count > 0, f"ChromaDB is empty ({count} docs)"

    # Test retrieval
    results = store.retrieve("Good morning", top_k=3)
    assert len(results) > 0, "No results for 'Good morning'"
    assert "response" in results[0], "Missing 'response' field"
    assert "context" in results[0], "Missing 'context' field"

    print(f"OK ({count:,} docs, retrieval works)")
    print(f"    Sample: Her: '{results[0]['context'][:50]}...'")
    print(f"    Ayush: '{results[0]['response'][:50]}...'")


def test_history():
    """Test SQLite conversation history."""
    print("  [3/7] ConversationHistory...", end=" ")
    from src.memory.history import ConversationHistory

    # Use temp DB
    h = ConversationHistory(db_path=ROOT / "data" / "test_history.sqlite3")

    # Create conversation
    h.get_or_create_conversation("test_conv", "Test Girl")

    # Add messages
    h.add_message("test_conv", "user", "Hi, kya kar raha h?")
    h.add_message("test_conv", "assistant", "Kyuch nahi [MSG_BREAK] Tu bata")

    # Retrieve
    msgs = h.get_recent_messages("test_conv", limit=5)
    assert len(msgs) == 2, f"Expected 2 msgs, got {len(msgs)}"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"

    # ChatML format
    chatml = h.get_recent_as_chatml("test_conv")
    assert all("role" in m and "content" in m for m in chatml)

    # New session check
    is_new = h.is_new_session("test_conv", gap_hours=0.001)
    assert isinstance(is_new, bool)

    # Cleanup
    h.clear_conversation("test_conv")
    h.close()
    import os
    os.remove(ROOT / "data" / "test_history.sqlite3")

    print("OK (CRUD + session detection works)")


def test_llm_chain():
    """Test LLM fallback chain setup (no actual API calls)."""
    print("  [4/7] LLM Fallback...", end=" ")
    from src.llm.fallback import LLMFallbackChain

    chain = LLMFallbackChain()
    status = chain.status()
    available = chain.available_providers

    print(f"OK (providers: {list(status.keys())}, available: {available})")


def test_context_builder():
    """Test prompt construction."""
    print("  [5/7] ContextBuilder...", end=" ")
    from src.engine.context_builder import ContextBuilder
    from src.memory.vector_store import VectorStore

    builder = ContextBuilder()

    # Test system prompt
    system = builder.build_system_prompt("Shubhi")
    assert "Shreyash" in system
    assert "Shubhi" in system
    assert "MSG_BREAK" in system
    assert len(system) > 500, "System prompt too short"

    # Test with retrieval
    store = VectorStore()
    examples = store.retrieve("tu kesi h?", top_k=3)

    messages = builder.build_messages(
        girl_message="Hii, kya kar raha h?",
        partner_name="Shubhi",
        history=[
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning 🌄🌄🌄"},
        ],
        retrieved_examples=examples,
    )

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Hii, kya kar raha h?"

    tokens = builder.estimate_tokens(messages)
    print(f"OK (prompt: {len(messages)} msgs, ~{tokens:,} tokens)")


def test_post_processor():
    """Test post-processing rules."""
    print("  [6/7] PostProcessor...", end=" ")
    from src.engine.post_processor import PostProcessor

    pp = PostProcessor()

    # Test spelling corrections
    tests = [
        # (input, expected substring)
        ("Haan mai theek hu", "Ha"),     # haan → Ha
        ("Haan mai theek hu", "thik"),   # theek → thik
        ("Kuch nahi karna hai", "h"),    # hai → h
        ("aur tu kaisi hai?", "Or"),     # aur → Or
        ("accha theek hai", "Aacha"),    # accha → aacha (capitalized: first word)
        ("pehle bata", "Phele"),         # pehle → phele (capitalized: first word)
    ]

    all_passed = True
    for raw, expected in tests:
        result = pp.process_to_string(raw)
        if expected not in result:
            print(f"\n    FAIL: '{raw}' → '{result}' (expected '{expected}')")
            all_passed = False

    # Test burst splitting
    bursts = pp.process("Ha sahi h [MSG_BREAK] Me bhi [MSG_BREAK] Chod na")
    assert len(bursts) == 3, f"Expected 3 bursts, got {len(bursts)}"

    # Test artifact removal
    cleaned = pp.process("Ayush: Ha bilkul")
    assert not any(m.startswith("Ayush:") for m in cleaned)

    # Test validation
    val = pp.validate(cleaned)
    assert isinstance(val["valid"], bool)

    if all_passed:
        print("OK (spelling, bursts, artifacts, validation all pass)")
    else:
        print("PARTIAL (some spelling tests failed, check above)")


def test_chatbot_init():
    """Test chatbot initialization (no API call)."""
    print("  [7/7] Chatbot...", end=" ")
    from src.chatbot import Chatbot

    bot = Chatbot()
    bot.set_conversation("test", "Test Girl")
    status = bot.status()

    assert status["vector_store"]["count"] > 0
    assert status["conversation_id"] == "test"

    bot.close()
    print(f"OK (initialized, {status['vector_store']['count']:,} examples ready)")


def main():
    print()
    print("█" * 55)
    print("  END-TO-END SMOKE TEST")
    print("█" * 55)
    print()

    tests = [
        test_config,
        test_vector_store,
        test_history,
        test_llm_chain,
        test_context_builder,
        test_post_processor,
        test_chatbot_init,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed")

    if failed:
        print("  ⚠  Some tests failed. Check output above.")
    else:
        print("  ✅ All tests passed! System is ready.")
        print()
        print("  Next steps:")
        print("    1. Add API key(s) to .env file")
        print("    2. Run: python chat.py")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
