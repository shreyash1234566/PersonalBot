"""
Style Analyzer
==============
Analyzes all "I Am All" messages to extract a comprehensive style bible:
- Word frequencies & spelling map
- Emoji usage profile
- Message length statistics
- Greeting/closing patterns
- Per-chat style variations

Input:  data/parsed/parsed_messages.jsonl
Output: config/style_bible.json
"""

import json
import re
import os
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import unicodedata

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "parsed" / "parsed_messages.jsonl"
OUTPUT_FILE = ROOT_DIR / "config" / "style_bible.json"

USER_SENDER = "I Am All"

# Emoji regex (covers most emoji ranges)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed chars
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U0000200B-\U0000200F"  # zero-width chars
    "\U00002028-\U0000202F"  # line/paragraph separators
    "\U0000205F-\U00002060"  # medium math space
    "\U0000231A-\U0000231B"  # watch, hourglass
    "\U000023E9-\U000023F3"  # media controls
    "\U000023F8-\U000023FA"  # more media
    "]+",
    flags=re.UNICODE,
)

# Known spelling map (Ayush-specific → standard Hinglish)
# This will be enriched by the analysis
KNOWN_SPELLING_PAIRS = {
    # Ayush's spelling: standard form
    "h": "hai",
    "nhi": "nahi",
    "Or": "aur",
    "Ha": "haan",
    "aacha": "accha",
    "phele": "pehle",
    "kyuch": "kuch",
    "thik": "theek",
    "to": "toh",
    "kesi": "kaisi",
    "hme": "humein",
    "kro": "karo",
    "krta": "karta",
    "krti": "karti",
    "krna": "karna",
    "krne": "karne",
    "kya": "kya",
    "bhi": "bhi",
    "me": "mein",
    "bol": "bol",
    "btao": "batao",
    "Ho": "ho",
    "jao": "jao",
    "aao": "aao",
    "sab": "sab",
    "rha": "raha",
    "rahi": "rahi",
    "rah": "raha",
    "isake": "iske",
    "teri": "teri",
    "meri": "meri",
    "Hmm": "hmm",
    "Ook": "ok",
    "Sahi": "sahi",
}

# Greeting patterns to detect
MORNING_RE = re.compile(r"good\s*morning", re.IGNORECASE)
NIGHT_RE = re.compile(r"good\s*night", re.IGNORECASE)


def load_user_messages(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load all messages, return (user_msgs, all_msgs)."""
    all_msgs = []
    user_msgs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            all_msgs.append(msg)
            if msg["sender"] == USER_SENDER:
                user_msgs.append(msg)
    return user_msgs, all_msgs


def extract_emojis(text: str) -> list[str]:
    """Extract individual emoji characters from text."""
    emojis = []
    for match in EMOJI_RE.finditer(text):
        # Split compound emoji matches into individual ones
        segment = match.group()
        for char in segment:
            if unicodedata.category(char).startswith(("So", "Sk")) or ord(char) > 0x1F000:
                emojis.append(char)
            elif char in "☺️☠️❤️⭐✨🌟💫⚡🔥":
                emojis.append(char)
    # Simpler approach: just find all emoji-like chars
    simple_emojis = []
    for char in text:
        if ord(char) > 0x1F000 or char in "☺☠❤⭐✨🌟💫⚡🔥♥️←→↓↑⬆⬇":
            simple_emojis.append(char)
    return simple_emojis if simple_emojis else emojis


def analyze_word_frequencies(messages: list[dict]) -> dict:
    """Get word frequency distribution."""
    word_counter = Counter()
    for msg in messages:
        text = msg["message"]
        # Remove emojis before tokenizing
        text = EMOJI_RE.sub("", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Tokenize
        words = re.findall(r"[A-Za-z\u0900-\u097F]+(?:'[A-Za-z]+)?", text)
        for word in words:
            word_counter[word] += 1
    
    return dict(word_counter.most_common(300))


def analyze_emoji_usage(messages: list[dict]) -> dict:
    """Get emoji frequency and context patterns."""
    emoji_counter = Counter()
    emoji_contexts = defaultdict(list)
    
    for msg in messages:
        text = msg["message"]
        emojis = extract_emojis(text)
        for emoji in emojis:
            emoji_counter[emoji] += 1
            # Get surrounding text as context
            clean_text = EMOJI_RE.sub("", text).strip()
            if clean_text and len(clean_text) < 100:
                emoji_contexts[emoji].append(clean_text)
    
    # Build emoji profile
    total_msgs = len(messages)
    emoji_msgs = sum(1 for m in messages if extract_emojis(m["message"]))
    
    top_emojis = []
    for emoji, count in emoji_counter.most_common(30):
        contexts = emoji_contexts[emoji][:5]  # Top 5 context examples
        top_emojis.append({
            "emoji": emoji,
            "count": count,
            "frequency_pct": round(count / total_msgs * 100, 2),
            "sample_contexts": contexts,
        })
    
    return {
        "total_messages": total_msgs,
        "messages_with_emojis": emoji_msgs,
        "emoji_usage_rate": round(emoji_msgs / total_msgs * 100, 2) if total_msgs else 0,
        "top_emojis": top_emojis,
    }


def analyze_message_lengths(messages: list[dict]) -> dict:
    """Compute message length statistics."""
    lengths = [len(m["message"]) for m in messages]
    word_counts = [len(m["message"].split()) for m in messages]
    
    if not lengths:
        return {}
    
    lengths_sorted = sorted(lengths)
    words_sorted = sorted(word_counts)
    n = len(lengths)
    
    return {
        "total_messages": n,
        "char_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / n, 1),
            "median": lengths_sorted[n // 2],
            "p25": lengths_sorted[n // 4],
            "p75": lengths_sorted[3 * n // 4],
            "p90": lengths_sorted[int(n * 0.9)],
        },
        "word_count": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(sum(word_counts) / n, 1),
            "median": words_sorted[n // 2],
            "p25": words_sorted[n // 4],
            "p75": words_sorted[3 * n // 4],
            "p90": words_sorted[int(n * 0.9)],
        },
        "length_buckets": {
            "1-10_chars": sum(1 for l in lengths if l <= 10),
            "11-30_chars": sum(1 for l in lengths if 11 <= l <= 30),
            "31-60_chars": sum(1 for l in lengths if 31 <= l <= 60),
            "61-100_chars": sum(1 for l in lengths if 61 <= l <= 100),
            "100+_chars": sum(1 for l in lengths if l > 100),
        },
    }


def analyze_greetings(messages: list[dict]) -> dict:
    """Extract greeting and closing patterns."""
    morning_patterns = Counter()
    night_patterns = Counter()
    
    for msg in messages:
        text = msg["message"].strip()
        if MORNING_RE.search(text):
            morning_patterns[text] += 1
        if NIGHT_RE.search(text):
            night_patterns[text] += 1
    
    return {
        "morning_greetings": [
            {"text": t, "count": c}
            for t, c in morning_patterns.most_common(10)
        ],
        "night_closings": [
            {"text": t, "count": c}
            for t, c in night_patterns.most_common(10)
        ],
    }


def analyze_response_patterns(messages: list[dict]) -> dict:
    """Analyze short response patterns (Ha, Hmm, Ok, etc.)."""
    short_responses = Counter()
    
    for msg in messages:
        text = msg["message"].strip()
        # Only count messages that are 1-3 words and < 20 chars
        words = text.split()
        if len(words) <= 3 and len(text) <= 20:
            short_responses[text] += 1
    
    return {
        "top_short_responses": [
            {"text": t, "count": c}
            for t, c in short_responses.most_common(50)
        ],
    }


def analyze_burst_patterns(all_msgs: list[dict]) -> dict:
    """Analyze how Ayush sends message bursts."""
    from datetime import timedelta
    
    # Group by chat
    by_chat = defaultdict(list)
    for msg in all_msgs:
        by_chat[msg["chat_id"]].append(msg)
    
    burst_lengths = []
    
    for chat_id, chat_msgs in by_chat.items():
        chat_msgs.sort(key=lambda x: x["timestamp"])
        
        current_burst = 0
        for i, msg in enumerate(chat_msgs):
            if msg["sender"] != USER_SENDER:
                if current_burst > 0:
                    burst_lengths.append(current_burst)
                current_burst = 0
                continue
            
            if current_burst == 0:
                current_burst = 1
            else:
                # Check time gap
                prev_ts = datetime.fromisoformat(chat_msgs[i - 1]["timestamp"])
                curr_ts = datetime.fromisoformat(msg["timestamp"])
                if curr_ts - prev_ts <= timedelta(seconds=60):
                    current_burst += 1
                else:
                    burst_lengths.append(current_burst)
                    current_burst = 1
        
        if current_burst > 0:
            burst_lengths.append(current_burst)
    
    if not burst_lengths:
        return {}
    
    burst_counter = Counter(burst_lengths)
    
    return {
        "total_bursts": len(burst_lengths),
        "avg_burst_length": round(sum(burst_lengths) / len(burst_lengths), 2),
        "max_burst_length": max(burst_lengths),
        "burst_size_distribution": {
            str(k): v for k, v in sorted(burst_counter.items())
            if k <= 10
        },
        "single_message_pct": round(
            burst_counter.get(1, 0) / len(burst_lengths) * 100, 1
        ),
        "multi_message_pct": round(
            sum(v for k, v in burst_counter.items() if k > 1)
            / len(burst_lengths) * 100,
            1,
        ),
    }


def analyze_time_patterns(messages: list[dict]) -> dict:
    """Analyze what hours Ayush is most active."""
    hour_counter = Counter()
    day_counter = Counter()
    
    for msg in messages:
        dt = datetime.fromisoformat(msg["timestamp"])
        hour_counter[dt.hour] += 1
        day_counter[dt.strftime("%A")] += 1
    
    return {
        "hourly_distribution": {
            f"{h:02d}:00": hour_counter.get(h, 0) for h in range(24)
        },
        "daily_distribution": dict(day_counter.most_common()),
        "peak_hours": [
            f"{h:02d}:00" for h, _ in hour_counter.most_common(5)
        ],
    }


def build_spelling_map(messages: list[dict]) -> dict:
    """
    Build Ayush's personal spelling map by analyzing most frequent words.
    Compares against standard Hinglish spellings.
    """
    word_freq = Counter()
    for msg in messages:
        text = EMOJI_RE.sub("", msg["message"])
        text = re.sub(r"https?://\S+", "", text)
        words = re.findall(r"[A-Za-z]+", text)
        for w in words:
            word_freq[w] += 1
    
    # Build the confirmed spelling map from actual data
    spelling_map = {}
    for ayush_spelling, standard in KNOWN_SPELLING_PAIRS.items():
        count = word_freq.get(ayush_spelling, 0)
        if count > 0:
            spelling_map[ayush_spelling] = {
                "standard_form": standard,
                "ayush_count": count,
                "confirmed": True,
            }
    
    return spelling_map


def per_chat_analysis(user_msgs: list[dict]) -> dict:
    """Compute style stats per chat partner."""
    by_chat = defaultdict(list)
    for msg in user_msgs:
        by_chat[msg["chat_id"]].append(msg)
    
    per_chat = {}
    for chat_id, msgs in by_chat.items():
        lengths = analyze_message_lengths(msgs)
        emojis = analyze_emoji_usage(msgs)
        greetings = analyze_greetings(msgs)
        responses = analyze_response_patterns(msgs)
        
        per_chat[chat_id] = {
            "message_count": len(msgs),
            "message_lengths": lengths,
            "emoji_profile": emojis,
            "greetings": greetings,
            "short_responses": responses,
        }
    
    return per_chat


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(input_file: Path = None, output_file: Path = None) -> dict:
    """Run the full style analysis pipeline."""
    src = Path(input_file) if input_file else INPUT_FILE
    dst = Path(output_file) if output_file else OUTPUT_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 3: Style Analyzer")
    print("=" * 60)
    print(f"\n  Input:  {src}")
    print(f"  Output: {dst}")
    
    # ── Load ──
    user_msgs, all_msgs = load_user_messages(src)
    print(f"\n  Total messages: {len(all_msgs):,}")
    print(f"  'I Am All' messages: {len(user_msgs):,}")
    
    # ── Analyze ──
    print("\n  Analyzing word frequencies...")
    word_freq = analyze_word_frequencies(user_msgs)
    
    print("  Analyzing emoji usage...")
    emoji_profile = analyze_emoji_usage(user_msgs)
    
    print("  Analyzing message lengths...")
    msg_lengths = analyze_message_lengths(user_msgs)
    
    print("  Analyzing greeting patterns...")
    greetings = analyze_greetings(user_msgs)
    
    print("  Analyzing short responses...")
    responses = analyze_response_patterns(user_msgs)
    
    print("  Analyzing burst patterns...")
    bursts = analyze_burst_patterns(all_msgs)
    
    print("  Analyzing time patterns...")
    time_patterns = analyze_time_patterns(user_msgs)
    
    print("  Building spelling map...")
    spelling_map = build_spelling_map(user_msgs)
    
    print("  Computing per-chat breakdowns...")
    per_chat = per_chat_analysis(user_msgs)
    
    # ── Assemble Style Bible ──
    style_bible = {
        "_meta": {
            "generated_from": str(src),
            "total_messages_analyzed": len(user_msgs),
            "user_identity": USER_SENDER,
            "real_name": "Shreyash",
        },
        "word_frequencies": word_freq,
        "spelling_map": spelling_map,
        "emoji_profile": emoji_profile,
        "message_lengths": msg_lengths,
        "greetings_and_closings": greetings,
        "short_responses": responses,
        "burst_patterns": bursts,
        "time_patterns": time_patterns,
        "per_chat_breakdown": per_chat,
    }
    
    # ── Save ──
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(style_bible, f, ensure_ascii=False, indent=2)
    
    # ── Summary ──
    print(f"\n  ── Style Bible Summary ──")
    print(f"  Top 10 words: {', '.join(list(word_freq.keys())[:10])}")
    
    if emoji_profile.get("top_emojis"):
        top_e = [e["emoji"] for e in emoji_profile["top_emojis"][:5]]
        print(f"  Top 5 emojis: {' '.join(top_e)}")
    
    print(f"  Emoji usage rate: {emoji_profile.get('emoji_usage_rate', 0)}%")
    
    if msg_lengths.get("char_length"):
        print(f"  Avg message length: {msg_lengths['char_length']['mean']} chars, "
              f"{msg_lengths['word_count']['mean']} words")
    
    if bursts:
        print(f"  Avg burst size: {bursts.get('avg_burst_length', 0)} messages")
        print(f"  Multi-message rate: {bursts.get('multi_message_pct', 0)}%")
    
    confirmed_spellings = {k: v["standard_form"] for k, v in spelling_map.items() if v["confirmed"]}
    print(f"  Confirmed spelling rules: {len(confirmed_spellings)}")
    top_5 = list(confirmed_spellings.items())[:5]
    for ayush, std in top_5:
        print(f"    '{ayush}' (not '{std}')")
    
    if time_patterns.get("peak_hours"):
        print(f"  Peak hours: {', '.join(time_patterns['peak_hours'][:3])}")
    
    for cid, data in per_chat.items():
        print(f"\n  [{cid}] {data['message_count']:,} messages")
    
    print(f"\n  ✓ Saved style bible to {dst}")
    
    return style_bible


if __name__ == "__main__":
    run()
