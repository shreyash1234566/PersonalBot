"""
WhatsApp Chat Parser
====================
Parses raw WhatsApp chat exports into structured JSONL.
Handles multi-line messages, system message filtering,
chat boundary detection, and deduplication of overlapping exports.

Input:  train.txt (41,890 lines — 2 chats + 1 duplicate region)
Output: data/parsed/parsed_messages.jsonl
"""

import re
import json
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT_DIR / "train.txt"
OUTPUT_DIR = ROOT_DIR / "data" / "parsed"
OUTPUT_FILE = OUTPUT_DIR / "parsed_messages.jsonl"

# Known sender names
USER_SENDER = "I Am All"
GIRL_1 = "Class Cr"
GIRL_2 = "shubha 2 Kritika, May"

# Boundary: Shubhi duplicate export starts at this line (approx)
DUPLICATE_REGION_START = 32925

# ─── Regex Patterns ──────────────────────────────────────────────────────────

# Matches: M/DD/YY, H:MM AM/PM - Sender: Message
MSG_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+?):\s(.*)',
    re.DOTALL
)

# Matches system messages: M/DD/YY, H:MM AM/PM - SystemText (no colon-split)
SYS_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s(.+)',
    re.DOTALL
)

# ─── Skip Filters ────────────────────────────────────────────────────────────

SKIP_SUBSTRINGS = [
    "messages and calls are end-to-end encrypted",
    "<media omitted>",
    "this message was deleted",
    "you deleted this message",
    "missed voice call",
    "missed video call",
    "changed their phone number",
    "changed the subject",
    "changed this group",
    "added you",
    "removed you",
    "left the group",
    "joined using this group",
    "created this group",
    "changed the group description",
    "changed the group icon",
    "turned on disappearing messages",
    "turned off disappearing messages",
    "message timer was",
    "security code changed",
    "is now an admin",
    "gif omitted",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "document omitted",
    "contact card omitted",
    "live location shared",
    "location:",
    "‎",  # WhatsApp invisible character (system msg marker)
    "null",
    "waiting for this message",
]


def should_skip(text: str) -> bool:
    """Return True if message should be filtered out."""
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    # Skip very short meaningless messages
    if lower in ("", "null"):
        return True
    for pattern in SKIP_SUBSTRINGS:
        if pattern in lower:
            return True
    return False


# ─── Timestamp Parsing ────────────────────────────────────────────────────────

def parse_timestamp(date_str: str, time_str: str) -> str | None:
    """Parse 'M/DD/YY, H:MM AM/PM' into ISO 8601 string."""
    for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return None


# ─── Chat ID Detection ───────────────────────────────────────────────────────

def detect_chat_id(sender: str, line_number: int) -> str:
    """Determine which conversation a message belongs to."""
    if sender == GIRL_1:
        return "class_cr"
    if sender == GIRL_2:
        return "shubhi"
    if sender == USER_SENDER:
        # User appears in both chats — determine by position
        # Class Cr chat ends ~line 21926, Shubhi starts ~line 21928
        return "class_cr" if line_number <= 21926 else "shubhi"
    return "unknown"


# ─── Main Parser ─────────────────────────────────────────────────────────────

def parse_whatsapp_file(filepath: str | Path) -> list[dict]:
    """
    Parse a WhatsApp chat export file into structured messages.
    
    Returns list of dicts with keys:
        timestamp, sender, message, chat_id, line_number, _is_dup
    """
    messages = []
    current_msg = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            
            # ── Try matching as a new sender message ──
            match = MSG_RE.match(line)
            if match:
                # Flush previous message
                if current_msg is not None and not should_skip(current_msg["message"]):
                    messages.append(current_msg)
                
                date_str, time_str, sender, text = match.groups()
                sender = sender.strip()
                ts = parse_timestamp(date_str, time_str)
                
                if ts:
                    current_msg = {
                        "timestamp": ts,
                        "sender": sender,
                        "message": text.strip(),
                        "chat_id": detect_chat_id(sender, line_num),
                        "line_number": line_num,
                        "_is_dup": line_num >= DUPLICATE_REGION_START,
                    }
                else:
                    current_msg = None
                continue
            
            # ── Try matching as a system message (no sender:) ──
            sys_match = SYS_RE.match(line)
            if sys_match:
                # Flush previous, skip this system line
                if current_msg is not None and not should_skip(current_msg["message"]):
                    messages.append(current_msg)
                current_msg = None
                continue
            
            # ── Continuation line (multi-line message) ──
            if current_msg is not None:
                stripped = line.strip()
                if stripped:
                    current_msg["message"] += "\n" + stripped
            # else: blank line between chats or trailing whitespace — ignore
    
    # Flush last message
    if current_msg is not None and not should_skip(current_msg["message"]):
        messages.append(current_msg)
    
    return messages


def deduplicate_shubhi(messages: list[dict]) -> list[dict]:
    """
    Remove duplicate Shubhi messages from the overlapping export region.
    
    Strategy:
    - Lines 21928–32924 = primary Shubhi export (Apr 20 – Oct 21, 2025)
    - Lines 32925–41890 = duplicate partial export (Jul 18 – Aug 24, 2025)
    - Build a set of (timestamp, sender, msg_prefix) from primary
    - Keep duplicate-region messages ONLY if they don't exist in primary
    """
    # Build primary key set
    primary_keys = set()
    for msg in messages:
        if msg["chat_id"] == "shubhi" and not msg["_is_dup"]:
            # Use first 100 chars of message to handle minor whitespace diffs
            key = (msg["timestamp"], msg["sender"], msg["message"][:100])
            primary_keys.add(key)
    
    # Filter: keep everything that's NOT a duplicate
    result = []
    dup_removed = 0
    dup_promoted = 0
    
    for msg in messages:
        if msg["_is_dup"]:
            key = (msg["timestamp"], msg["sender"], msg["message"][:100])
            if key in primary_keys:
                dup_removed += 1
            else:
                # Unique message only in duplicate region — promote it
                msg["_is_dup"] = False
                msg["chat_id"] = "shubhi"
                result.append(msg)
                dup_promoted += 1
        else:
            result.append(msg)
    
    # Sort by timestamp, then line_number for stable ordering
    result.sort(key=lambda x: (x["timestamp"], x["line_number"]))
    
    print(f"  Duplicates removed: {dup_removed}")
    print(f"  Unique msgs promoted from dup region: {dup_promoted}")
    
    return result


def clean_for_output(messages: list[dict]) -> list[dict]:
    """Remove internal fields before saving."""
    cleaned = []
    for msg in messages:
        out = {
            "timestamp": msg["timestamp"],
            "sender": msg["sender"],
            "message": msg["message"],
            "chat_id": msg["chat_id"],
            "line_number": msg["line_number"],
        }
        cleaned.append(out)
    return cleaned


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(input_file: str | Path = None, output_file: str | Path = None) -> list[dict]:
    """Run the full parser pipeline. Returns cleaned message list."""
    src = Path(input_file) if input_file else RAW_FILE
    dst = Path(output_file) if output_file else OUTPUT_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: WhatsApp Chat Parser")
    print("=" * 60)
    print(f"\n  Input:  {src}")
    print(f"  Output: {dst}")
    
    # ── Parse ──
    print(f"\n  Parsing...")
    messages = parse_whatsapp_file(src)
    print(f"  Raw messages extracted: {len(messages):,}")
    
    # ── Pre-dedup stats ──
    cc = sum(1 for m in messages if m["chat_id"] == "class_cr")
    sp = sum(1 for m in messages if m["chat_id"] == "shubhi" and not m["_is_dup"])
    sd = sum(1 for m in messages if m["_is_dup"])
    
    print(f"\n  Class Cr messages:        {cc:,}")
    print(f"  Shubhi primary messages:  {sp:,}")
    print(f"  Shubhi duplicate region:  {sd:,}")
    
    # ── Deduplicate ──
    print(f"\n  Deduplicating Shubhi overlap...")
    messages = deduplicate_shubhi(messages)
    print(f"  Messages after dedup: {len(messages):,}")
    
    # ── Final stats ──
    sender_counts = defaultdict(int)
    chat_counts = defaultdict(int)
    for m in messages:
        sender_counts[m["sender"]] += 1
        chat_counts[m["chat_id"]] += 1
    
    print(f"\n  Sender breakdown:")
    for sender, count in sorted(sender_counts.items(), key=lambda x: -x[1]):
        print(f"    {sender}: {count:,}")
    
    print(f"\n  Chat breakdown:")
    for cid, count in sorted(chat_counts.items()):
        print(f"    {cid}: {count:,}")
    
    iam = sender_counts.get(USER_SENDER, 0)
    print(f"\n  '{USER_SENDER}' total messages: {iam:,}")
    
    # ── Save ──
    cleaned = clean_for_output(messages)
    with open(dst, "w", encoding="utf-8") as f:
        for msg in cleaned:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    
    print(f"\n  ✓ Saved {len(cleaned):,} messages to {dst}")
    return cleaned


if __name__ == "__main__":
    run()
