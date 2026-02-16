"""
Example Bank Generator
======================
Creates a categorized, searchable example bank from conversation sessions.
Each entry maps a girl's message (query) → Ayush's reply (response) with
category tags for retrieval-augmented generation.

Input:  data/parsed/parsed_messages.jsonl
Output: data/examples/example_bank.jsonl
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "parsed" / "parsed_messages.jsonl"
OUTPUT_FILE = ROOT_DIR / "data" / "examples" / "example_bank.jsonl"

USER_SENDER = "I Am All"

# Burst window for merging consecutive messages
BURST_WINDOW = timedelta(seconds=60)

# Session gap
SESSION_GAP = timedelta(hours=2)

# ─── Category Detection ──────────────────────────────────────────────────────

CATEGORY_PATTERNS = {
    "greeting_morning": [
        re.compile(r"good\s*morning", re.I),
        re.compile(r"subah|subh|suprabhat", re.I),
    ],
    "greeting_night": [
        re.compile(r"good\s*night", re.I),
        re.compile(r"shubh\s*ratri", re.I),
    ],
    "greeting_general": [
        re.compile(r"^(hi|hello|hey|hii+|hlo|helloo+)\b", re.I),
        re.compile(r"^(kya\s*hal|kesi\s*ho|kaisi\s*ho|kaise\s*ho)", re.I),
    ],
    "flirting": [
        re.compile(r"(pyar|love|crush|dil|heart|ishq|pati|wife|husband|shaadi|sindur)", re.I),
        re.compile(r"(cute|beautiful|pretty|hot|sexy|handsome|acch[ai]\s*lag)", re.I),
        re.compile(r"(miss\s*(you|kar)|yaad\s*aa)", re.I),
        re.compile(r"(date|propose|relationship)", re.I),
    ],
    "teasing": [
        re.compile(r"(pagal|bewakoof|stupid|idiot|buddhu|nalayak|chapri)", re.I),
        re.compile(r"(mazak|joke|funny|haha|lol|rofl|😂|🤣|😈)", re.I),
        re.compile(r"(chup|hatt|ja\s*na|dur\s*ho)", re.I),
    ],
    "emotional_support": [
        re.compile(r"(sad|upset|cry|ro\s|rona|dukhi|tension|stress|depres)", re.I),
        re.compile(r"(problem|dikkat|pareshani|worried|anxious)", re.I),
        re.compile(r"(hospital|bimar|sick|health|doctor)", re.I),
        re.compile(r"(care|samjh|understand|support|help\s*kar)", re.I),
    ],
    "planning": [
        re.compile(r"(milte|milna|meet|plan|chalte|chalo|aaja|aao)", re.I),
        re.compile(r"(kal|tomorrow|weekend|sunday|saturday)", re.I),
        re.compile(r"(movie|outing|trip|cafe|restaurant)", re.I),
    ],
    "gaming": [
        re.compile(r"(bgmi|pubg|game|gaming|match|rank|push|classic|tdm)", re.I),
        re.compile(r"(chicken\s*dinner|squad|duo|solo|drop)", re.I),
    ],
    "daily_update": [
        re.compile(r"(kya\s*kar\s*r[ah]|kya\s*ho\s*r[ah]|kya\s*chal\s*r[ah])", re.I),
        re.compile(r"(class|college|school|office|work|assignment|exam|test)", re.I),
        re.compile(r"(khana|breakfast|lunch|dinner|soya|utha|neend)", re.I),
    ],
    "compliment": [
        re.compile(r"(ach+[ai]\s*(h|ho|lag)|nice|great|amazing|awesome)", re.I),
        re.compile(r"(smart|intelligent|talented|best)", re.I),
    ],
    "argument": [
        re.compile(r"(gussa|angry|naraz|fight|ladai|jhagda)", re.I),
        re.compile(r"(sorry|maaf|galti|mistake)", re.I),
        re.compile(r"(block|ignore|seen|reply\s*nahi|baat\s*nahi)", re.I),
    ],
    "philosophy": [
        re.compile(r"(life|zindagi|destiny|kismat|bhagwan|god)", re.I),
        re.compile(r"(future|career|dream|goal|success)", re.I),
        re.compile(r"(believe|sochta|think|opinion|perspective)", re.I),
    ],
    "media_reaction": [
        re.compile(r"(song|gaana|movie|film|show|series|reel|meme)", re.I),
        re.compile(r"(insta|instagram|youtube|yt|spotify)", re.I),
    ],
}


def detect_categories(text: str) -> list[str]:
    """Detect all matching categories for a message."""
    categories = []
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                categories.append(category)
                break  # One match per category is enough
    
    if not categories:
        categories = ["general"]
    
    return categories


# ─── Message Processing ──────────────────────────────────────────────────────

def load_messages(filepath: Path) -> list[dict]:
    """Load parsed messages from JSONL."""
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def segment_sessions(messages: list[dict]) -> list[list[dict]]:
    """Split messages into sessions by SESSION_GAP."""
    if not messages:
        return []
    
    sessions = []
    current = [messages[0]]
    
    for msg in messages[1:]:
        prev_t = datetime.fromisoformat(current[-1]["timestamp"])
        curr_t = datetime.fromisoformat(msg["timestamp"])
        
        if curr_t - prev_t > SESSION_GAP:
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)
    
    if current:
        sessions.append(current)
    
    return sessions


def merge_consecutive_sender(messages: list[dict], sender: str) -> list[dict]:
    """Merge consecutive messages from `sender` within BURST_WINDOW using [MSG_BREAK]."""
    if not messages:
        return []
    
    merged = []
    buf = None
    
    for msg in messages:
        if msg["sender"] == sender:
            if buf and buf["sender"] == sender:
                prev_t = datetime.fromisoformat(buf["timestamp"])
                curr_t = datetime.fromisoformat(msg["timestamp"])
                if curr_t - prev_t <= BURST_WINDOW:
                    buf["message"] += " [MSG_BREAK] " + msg["message"]
                    continue
            # Start new buffer
            if buf:
                merged.append(buf)
            buf = dict(msg)  # copy
        else:
            if buf:
                merged.append(buf)
                buf = None
            merged.append(msg)
    
    if buf:
        merged.append(buf)
    
    return merged


def extract_examples_from_session(
    session: list[dict],
    chat_id: str,
) -> list[dict]:
    """
    Extract (girl_message → ayush_reply) pairs from a session.
    
    Each example includes:
    - context: the girl's message(s) that prompted the reply
    - response: Ayush's reply (with [MSG_BREAK] for bursts)
    - preceding_context: up to 5 prior exchanges for richer context
    - categories: auto-detected from both messages
    """
    # Merge Ayush's bursts
    merged = merge_consecutive_sender(session, USER_SENDER)
    
    examples = []
    
    for i, msg in enumerate(merged):
        if msg["sender"] != USER_SENDER:
            continue
        
        # Find the girl's message(s) immediately before this reply
        girl_context = []
        j = i - 1
        while j >= 0 and merged[j]["sender"] != USER_SENDER:
            girl_context.insert(0, merged[j]["message"])
            j -= 1
        
        if not girl_context:
            continue  # Ayush replied without a preceding girl message (unlikely but skip)
        
        # Build preceding context (up to 5 exchanges before the girl's message)
        preceding = []
        start = max(0, j - 4)  # j is the last Ayush msg before girl's context
        for k in range(start, max(0, j + 1)):
            role = "assistant" if merged[k]["sender"] == USER_SENDER else "user"
            preceding.append({"role": role, "content": merged[k]["message"]})
        
        # Detect categories from both girl and Ayush messages
        combined_text = " ".join(girl_context) + " " + msg["message"]
        categories = detect_categories(combined_text)
        
        example = {
            "context": " [MSG_BREAK] ".join(girl_context),
            "response": msg["message"],
            "categories": categories,
            "chat_id": chat_id,
            "timestamp": msg["timestamp"],
            "preceding_context": preceding,
            "context_length": len(girl_context),
        }
        examples.append(example)
    
    return examples


# ─── Quality Filters ─────────────────────────────────────────────────────────

def quality_filter(examples: list[dict]) -> list[dict]:
    """Remove low-quality examples."""
    filtered = []
    
    for ex in examples:
        # Skip if response is too short and meaningless
        resp = ex["response"].replace("[MSG_BREAK]", "").strip()
        if len(resp) < 2:
            continue
        
        # Skip if context is empty
        if not ex["context"].strip():
            continue
        
        # Skip pure "Ha" / "Hmm" / "Ok" chains without substance
        # (keep them if they have preceding context — they're valid short responses)
        words = resp.split()
        boring_words = {"ha", "hmm", "ok", "ook", "hm", "mm"}
        if all(w.lower() in boring_words for w in words) and not ex["preceding_context"]:
            continue
        
        filtered.append(ex)
    
    return filtered


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(input_file: Path = None, output_file: Path = None) -> list[dict]:
    """Run the example bank generation pipeline."""
    src = Path(input_file) if input_file else INPUT_FILE
    dst = Path(output_file) if output_file else OUTPUT_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 4: Example Bank Generator")
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
    
    all_examples = []
    
    for chat_id, chat_msgs in by_chat.items():
        chat_msgs.sort(key=lambda x: x["timestamp"])
        sessions = segment_sessions(chat_msgs)
        
        chat_examples = []
        for session in sessions:
            if len(session) < 3:
                continue
            examples = extract_examples_from_session(session, chat_id)
            chat_examples.extend(examples)
        
        # Apply quality filter
        before = len(chat_examples)
        chat_examples = quality_filter(chat_examples)
        after = len(chat_examples)
        
        print(f"\n  [{chat_id}]")
        print(f"    Sessions: {len(sessions):,}")
        print(f"    Raw examples: {before:,}")
        print(f"    After quality filter: {after:,}")
        
        all_examples.extend(chat_examples)
    
    # ── Category distribution ──
    cat_counter = Counter()
    for ex in all_examples:
        for cat in ex["categories"]:
            cat_counter[cat] += 1
    
    print(f"\n  Category distribution:")
    for cat, count in cat_counter.most_common():
        print(f"    {cat}: {count:,}")
    
    # ── Save ──
    with open(dst, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\n  Total examples: {len(all_examples):,}")
    print(f"  ✓ Saved to {dst}")
    
    return all_examples


if __name__ == "__main__":
    run()
