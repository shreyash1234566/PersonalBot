"""
Session Builder
===============
Groups parsed messages into conversation sessions, maps roles to ChatML format,
merges burst messages with [MSG_BREAK], and outputs training-ready JSONL.

Input:  data/parsed/parsed_messages.jsonl
Output: data/sessions/conversations.jsonl
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "parsed" / "parsed_messages.jsonl"
OUTPUT_DIR = ROOT_DIR / "data" / "sessions"
OUTPUT_FILE = OUTPUT_DIR / "conversations.jsonl"

USER_SENDER = "I Am All"

# Session gap: 2 hours of silence = new session
SESSION_GAP = timedelta(hours=2)

# Burst window: consecutive messages from same sender within 60s = one logical message
BURST_WINDOW = timedelta(seconds=60)

# Minimum messages in a session to be useful for training
MIN_SESSION_MESSAGES = 3

# Context window: how many prior messages to include as context
CONTEXT_WINDOW = 10

# System prompt template (will be filled with style bible data later)
SYSTEM_PROMPT_TEMPLATE = """You are Ayush ("I Am All"), a Hinglish-speaking guy chatting with a girl. You must perfectly replicate Ayush's texting style:

CORE RULES:
- Write in Hinglish (Hindi in Roman script mixed with English)
- Use Ayush's exact spelling: "h" not "hai", "Or" not "aur", "nhi" not "nahi", "Ha" not "haan", "aacha" not "accha", "phele" not "pehle", "kyuch" not "kuch", "to" not "toh", "thik" not "theek"
- Send SHORT messages (5-15 words typically). Use [MSG_BREAK] between separate messages in a burst
- Capitalize first word of sentences, keep rest lowercase
- Primary emojis: 😅 (most used), 🌄 (morning), 🌉 (night), 😈 (flirty), 🤧, 🙃, 😂
- Short acknowledgments: "Ha", "Hmm", "Ook", "Aacha", "Sahi h"
- Morning greeting: "Good morning 🌄🌄🌄" or "Good morning 🌄"
- Night closing: "Good night 🌉🌉🌉"
- Teasing/playful tone, never dry or overly formal
- Mirror the girl's energy: if she's playful, be more playful; if serious, add light humor but show care
- Use rapid-fire short messages, NOT one long paragraph

PERSONALITY:
- Witty, slightly sarcastic humor
- Protective but not possessive
- Uses humor to navigate serious topics
- References: BGMI, coding, friends, college life
- Never uses "ji", "aapka", or overly formal Hindi

CHAT PARTNER: {partner_name}
RELATIONSHIP: {relationship_type}"""


def load_messages(filepath: Path) -> list[dict]:
    """Load parsed messages from JSONL."""
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def parse_iso_timestamp(ts: str) -> datetime:
    """Parse ISO 8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts)


def segment_sessions(messages: list[dict]) -> list[list[dict]]:
    """Split messages into sessions based on SESSION_GAP."""
    if not messages:
        return []
    
    sessions = []
    current_session = [messages[0]]
    
    for msg in messages[1:]:
        prev_time = parse_iso_timestamp(current_session[-1]["timestamp"])
        curr_time = parse_iso_timestamp(msg["timestamp"])
        
        if curr_time - prev_time > SESSION_GAP:
            sessions.append(current_session)
            current_session = [msg]
        else:
            current_session.append(msg)
    
    if current_session:
        sessions.append(current_session)
    
    return sessions


def merge_bursts(messages: list[dict]) -> list[dict]:
    """
    Merge consecutive messages from the same sender within BURST_WINDOW
    into a single message separated by [MSG_BREAK].
    """
    if not messages:
        return []
    
    merged = []
    current = {
        "timestamp": messages[0]["timestamp"],
        "sender": messages[0]["sender"],
        "message": messages[0]["message"],
        "chat_id": messages[0]["chat_id"],
    }
    
    for msg in messages[1:]:
        same_sender = msg["sender"] == current["sender"]
        prev_time = parse_iso_timestamp(current["timestamp"])
        curr_time = parse_iso_timestamp(msg["timestamp"])
        within_burst = (curr_time - prev_time) <= BURST_WINDOW
        
        if same_sender and within_burst:
            # Merge into current burst
            current["message"] += " [MSG_BREAK] " + msg["message"]
        else:
            merged.append(current)
            current = {
                "timestamp": msg["timestamp"],
                "sender": msg["sender"],
                "message": msg["message"],
                "chat_id": msg["chat_id"],
            }
    
    merged.append(current)
    return merged


def build_chatml_conversations(
    sessions: list[list[dict]], 
    chat_id: str
) -> list[dict]:
    """
    Convert sessions into ChatML training format.
    
    For each "I Am All" response, we create a training example:
    - system: style prompt
    - user: the girl's message(s) as context
    - assistant: Ayush's response
    
    Uses sliding window for multi-turn context.
    """
    partner_map = {
        "class_cr": ("Class Cr", "casual college friend, light banter"),
        "shubhi": ("Shubhi", "romantic interest, deeper emotional connection"),
    }
    
    partner_name, rel_type = partner_map.get(chat_id, ("Unknown", "unknown"))
    
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(
        partner_name=partner_name,
        relationship_type=rel_type,
    )
    
    conversations = []
    
    for session in sessions:
        # Merge bursts first
        merged = merge_bursts(session)
        
        if len(merged) < MIN_SESSION_MESSAGES:
            continue
        
        # Build multi-turn conversation
        turns = []
        for msg in merged:
            role = "assistant" if msg["sender"] == USER_SENDER else "user"
            turns.append({"role": role, "content": msg["message"]})
        
        # Skip sessions where Ayush never speaks
        if not any(t["role"] == "assistant" for t in turns):
            continue
        
        # Skip sessions where girl never speaks
        if not any(t["role"] == "user" for t in turns):
            continue
        
        # ── Sliding window approach ──
        # Create training examples for each assistant response
        for i, turn in enumerate(turns):
            if turn["role"] != "assistant":
                continue
            
            # Get context: up to CONTEXT_WINDOW prior messages
            start_idx = max(0, i - CONTEXT_WINDOW)
            context_turns = turns[start_idx:i]
            
            # Must have at least one user message in context
            if not any(t["role"] == "user" for t in context_turns):
                continue
            
            # Build the ChatML conversation
            chatml = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    *context_turns,
                    {"role": "assistant", "content": turn["content"]},
                ],
                "metadata": {
                    "chat_id": chat_id,
                    "partner": partner_name,
                    "timestamp": merged[min(i, len(merged) - 1)].get("timestamp", ""),
                    "session_length": len(merged),
                    "context_depth": len(context_turns),
                },
            }
            conversations.append(chatml)
        
        # ── Full session format (for few-shot retrieval) ──
        full_session = {
            "messages": [
                {"role": "system", "content": system_msg},
                *turns,
            ],
            "metadata": {
                "chat_id": chat_id,
                "partner": partner_name,
                "timestamp": merged[0].get("timestamp", ""),
                "session_length": len(merged),
                "format": "full_session",
            },
        }
        conversations.append(full_session)
    
    return conversations


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(input_file: Path = None, output_file: Path = None) -> list[dict]:
    """Run the session builder pipeline."""
    src = Path(input_file) if input_file else INPUT_FILE
    dst = Path(output_file) if output_file else OUTPUT_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 2: Session Builder")
    print("=" * 60)
    print(f"\n  Input:  {src}")
    print(f"  Output: {dst}")
    
    # ── Load ──
    messages = load_messages(src)
    print(f"\n  Loaded {len(messages):,} messages")
    
    # ── Group by chat ──
    by_chat = defaultdict(list)
    for msg in messages:
        by_chat[msg["chat_id"]].append(msg)
    
    all_conversations = []
    total_sessions = 0
    
    for chat_id, chat_msgs in by_chat.items():
        # Sort by timestamp
        chat_msgs.sort(key=lambda x: x["timestamp"])
        
        # Segment into sessions
        sessions = segment_sessions(chat_msgs)
        total_sessions += len(sessions)
        
        print(f"\n  [{chat_id}]")
        print(f"    Messages: {len(chat_msgs):,}")
        print(f"    Sessions: {len(sessions):,}")
        
        # Session length stats
        lengths = [len(s) for s in sessions]
        if lengths:
            print(f"    Session sizes: min={min(lengths)}, max={max(lengths)}, "
                  f"avg={sum(lengths)/len(lengths):.1f}")
        
        # Build ChatML
        convos = build_chatml_conversations(sessions, chat_id)
        all_conversations.extend(convos)
        
        sliding = sum(1 for c in convos if c["metadata"].get("format") != "full_session")
        full = sum(1 for c in convos if c["metadata"].get("format") == "full_session")
        print(f"    Training examples (sliding window): {sliding:,}")
        print(f"    Training examples (full session):   {full:,}")
    
    # ── Save ──
    with open(dst, "w", encoding="utf-8") as f:
        for convo in all_conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")
    
    print(f"\n  Total sessions across all chats: {total_sessions:,}")
    print(f"  Total training examples: {len(all_conversations):,}")
    print(f"  ✓ Saved to {dst}")
    
    return all_conversations


if __name__ == "__main__":
    run()
