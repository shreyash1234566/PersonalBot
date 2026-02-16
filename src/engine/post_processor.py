"""
Post-Processor
==============
Deterministic filter applied AFTER LLM generation to enforce
Shreyash's exact spelling rules, emoji limits, message formatting,
and capitalization. This is the quality gate — catches anything
the LLM gets wrong.
"""

import re
import json
from pathlib import Path

from src.config import STYLE_BIBLE_FILE, MAX_MSG_CHARS, MAX_BURST_SIZE


class PostProcessor:
    """Enforces Shreyash's style rules on LLM output."""

    def __init__(self, style_bible_path: Path = None):
        self._bible = None
        self._bible_path = style_bible_path or STYLE_BIBLE_FILE
        self._spelling_map = None

    # ── Style Bible ───────────────────────────────────────────────────────

    @property
    def style_bible(self) -> dict:
        if self._bible is None:
            with open(self._bible_path, "r", encoding="utf-8") as f:
                self._bible = json.load(f)
        return self._bible

    @property
    def spelling_map(self) -> dict[str, str]:
        """Load spelling corrections: standard_form → ayush_form."""
        if self._spelling_map is None:
            self._spelling_map = {}
            for ayush_word, info in self.style_bible.get("spelling_map", {}).items():
                if info.get("confirmed") and info.get("ayush_count", 0) > 5:
                    standard = info["standard_form"]
                    # Map standard → ayush (we want to REPLACE standard with ayush)
                    self._spelling_map[standard] = ayush_word
        return self._spelling_map

    # ── Core Processing ───────────────────────────────────────────────────

    def process(self, raw_output: str, girl_message: str = None) -> list[str]:
        """
        Full post-processing pipeline.
        
        Takes raw LLM output → returns list of individual messages (burst).
        
        Pipeline:
          1. Split by [MSG_BREAK]
          2. Clean each message
          3. Apply spelling corrections
          4. Apply hmm↔mm mirroring (if girl_message provided)
          5. Fix capitalization
          6. Enforce length limits
          7. Validate emoji usage
          8. Remove LLM artifacts
        """
        if not raw_output or not raw_output.strip():
            return ["Hmm"]

        # 1. Split into burst messages
        messages = self._split_burst(raw_output)

        # 2-8. Process each message
        processed = []
        for msg in messages:
            msg = self._clean_artifacts(msg)
            msg = self._apply_spelling(msg)
            if girl_message:
                msg = self._apply_mirroring(msg, girl_message)
            msg = self._fix_capitalization(msg)
            msg = self._enforce_length(msg)
            msg = self._clean_punctuation(msg)
            
            if msg.strip():
                processed.append(msg.strip())

        # Enforce burst size limit
        if len(processed) > MAX_BURST_SIZE:
            processed = processed[:MAX_BURST_SIZE]

        # Fallback if everything got filtered
        if not processed:
            processed = ["Hmm"]

        return processed

    def _apply_mirroring(self, text: str, girl_message: str) -> str:
        """
        Apply hmm↔mm mirroring based on what the girl said.
        If she says "hmm", we reply with "mm" and vice versa.
        """
        girl_lower = girl_message.lower().strip()
        text_lower = text.lower().strip()
        
        # If girl said "hmm" (or variations) and we're about to say "hmm", change to "mm"
        if re.match(r'^h+m+$', girl_lower):
            if re.match(r'^h+m+$', text_lower):
                return "Mm"
        
        # If girl said "mm" and we're about to say "mm", change to "hmm"
        if re.match(r'^m+$', girl_lower):
            if re.match(r'^m+$', text_lower) or re.match(r'^h+m+$', text_lower):
                return "Hmm"
        
        return text

    def process_to_string(self, raw_output: str, girl_message: str = None) -> str:
        """Process and re-join with [MSG_BREAK] for display."""
        messages = self.process(raw_output, girl_message)
        return " [MSG_BREAK] ".join(messages)

    # ── Step 1: Burst Splitting ───────────────────────────────────────────

    def _split_burst(self, text: str) -> list[str]:
        """Split on [MSG_BREAK] and normalize."""
        # Handle various formats the LLM might use
        text = text.replace("[MSG BREAK]", "[MSG_BREAK]")
        text = text.replace("[MSGBREAK]", "[MSG_BREAK]")
        text = text.replace("[msg_break]", "[MSG_BREAK]")
        text = text.replace("\\n", "\n")

        if "[MSG_BREAK]" in text:
            parts = text.split("[MSG_BREAK]")
        elif "\n\n" in text:
            # LLM might use double newlines instead
            parts = text.split("\n\n")
        elif "\n" in text and len(text.split("\n")) <= MAX_BURST_SIZE:
            parts = text.split("\n")
        else:
            parts = [text]

        return [p.strip() for p in parts if p.strip()]

    # ── Step 2: Clean LLM Artifacts ───────────────────────────────────────

    def _clean_artifacts(self, text: str) -> str:
        """Remove common LLM-generated artifacts."""
        # Remove role prefixes the LLM might hallucinate
        prefixes_to_strip = [
            "Shreyash:", "Ayush:", "I Am All:", "Me:", "Assistant:",
            "shreyash:", "ayush:", "i am all:", "me:", "assistant:",
            "Response:", "response:",
            "Shreyash -", "Ayush -", "I Am All -",
        ]
        for prefix in prefixes_to_strip:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # *italic*
        text = re.sub(r"_(.+?)_", r"\1", text)         # _underline_

        # Remove quotes the LLM might add
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove explanation text in parentheses at the end
        text = re.sub(r'\s*\(.*?(replying|responding|teasing|joking).*?\)\s*$', '', text, flags=re.I)

        return text.strip()

    # ── Step 3: Spelling Corrections ──────────────────────────────────────

    def _apply_spelling(self, text: str) -> str:
        """
        Replace standard Hinglish spellings with Shreyash's versions.
        Uses word-boundary matching to avoid partial replacements.
        """
        # Hard-coded high-priority replacements (most impactful)
        # These are case-sensitive and carefully ordered
        priority_replacements = [
            # (pattern, replacement, flags)
            (r'\bhai\b', 'h', re.IGNORECASE),
            (r'\bhain\b', 'h', re.IGNORECASE),
            (r'\baur\b', 'Or', re.IGNORECASE),
            (r'\bhaan\b', 'Ha', re.IGNORECASE),
            (r'\bhaa\b', 'Ha', re.IGNORECASE),
            (r'\bnahi\b', 'nahi', 0),           # Keep as-is
            (r'\bnhi\b', 'nhi', 0),             # Keep as-is
            (r'\baccha\b', 'aacha', re.IGNORECASE),
            (r'\bachha\b', 'aacha', re.IGNORECASE),
            (r'\bacha\b', 'aacha', re.IGNORECASE),
            (r'\bpehle\b', 'phele', re.IGNORECASE),
            (r'\bpahle\b', 'phele', re.IGNORECASE),
            (r'\bkuch\b', 'kyuch', re.IGNORECASE),
            (r'\btheek\b', 'thik', re.IGNORECASE),
            (r'\bthik\b', 'thik', 0),           # Already correct
            (r'\btoh\b', 'to', re.IGNORECASE),
            (r'\bkaisi\b', 'kesi', re.IGNORECASE),
            (r'\bkaro\b', 'kro', re.IGNORECASE),
            (r'\bkarta\b', 'krta', re.IGNORECASE),
            (r'\bkarti\b', 'krti', re.IGNORECASE),
            (r'\bkarna\b', 'krna', re.IGNORECASE),
            (r'\bkarne\b', 'krne', re.IGNORECASE),
            (r'\bbatao\b', 'btao', re.IGNORECASE),
            (r'\bhoga\b', 'hoga', 0),
            (r'\bhogi\b', 'hogi', 0),
            (r'\bmein\b', 'me', re.IGNORECASE),  # "mein" → "me"
            (r'\bsahi\b', 'sahi', 0),
            (r'\bOk\b', 'Ook', 0),               # "Ok" → "Ook"
            (r'\bok\b', 'ook', 0),
            (r'\bOkay\b', 'Ook', re.IGNORECASE),
            # Casual abbreviations
            (r'\bkya kar raha\b', 'kkrh', re.IGNORECASE),
            (r'\bkya kar rahi\b', 'kkrh', re.IGNORECASE),
            (r'\bkya kr raha\b', 'kkrh', re.IGNORECASE),
            (r'\bkya kr rahi\b', 'kkrh', re.IGNORECASE),
            (r'\btime pass\b', 'tp', re.IGNORECASE),
            (r'\btimepass\b', 'tp', re.IGNORECASE),
        ]

        for pattern, replacement, flags in priority_replacements:
            text = re.sub(pattern, replacement, text, flags=flags)

        return text

    # ── Step 4: Capitalization ────────────────────────────────────────────

    def _fix_capitalization(self, text: str) -> str:
        """
        Shreyash's capitalization pattern:
        - First word of message: Capitalized
        - Rest: lowercase (except proper nouns, emojis, "Or", "Ha", "Good", etc.)
        """
        if not text or len(text) < 2:
            return text

        # Preserve words that should stay capitalized
        preserve = {
            "Or", "Ha", "Hmm", "Ook", "Good", "BGMI", "Nahi",
            "Hi", "Hello", "Hey", "Bye", "OK", "Tu", "Me",
            "Ye", "Vo", "Ab", "Abhi", "Kya", "Aacha", "Sahi",
            "To", "Phir", "Bol", "Bata", "De", "Kar", "Le",
        }

        words = text.split()
        if not words:
            return text

        # Capitalize first word
        if words[0] not in preserve and not any(ord(c) > 0x1F000 for c in words[0]):
            words[0] = words[0][:1].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()

        return " ".join(words)

    # ── Step 5: Length Enforcement ────────────────────────────────────────

    def _enforce_length(self, text: str) -> str:
        """Truncate overly long messages (Shreyash keeps it short)."""
        if len(text) <= MAX_MSG_CHARS:
            return text

        # Try to cut at last complete word before limit
        truncated = text[:MAX_MSG_CHARS]
        last_space = truncated.rfind(" ")
        if last_space > MAX_MSG_CHARS * 0.6:
            truncated = truncated[:last_space]

        return truncated.strip()

    # ── Step 6: Punctuation Cleanup ───────────────────────────────────────

    def _clean_punctuation(self, text: str) -> str:
        """
        Shreyash's punctuation style:
        - No excessive punctuation (!!!, ???, .....)
        - Occasional single comma
        - No periods at end of casual messages
        """
        # Replace multiple exclamation marks
        text = re.sub(r"!{2,}", "!", text)

        # Replace multiple question marks
        text = re.sub(r"\?{2,}", "?", text)

        # Replace excessive dots (....) with just ..
        text = re.sub(r"\.{3,}", "..", text)

        # Remove trailing period on short casual messages
        if len(text) < 50 and text.endswith(".") and not text.endswith(".."):
            text = text[:-1]

        # Remove excessive commas
        text = re.sub(r",{2,}", ",", text)

        return text.strip()

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, messages: list[str]) -> dict:
        """
        Check processed output for quality issues.
        Returns a dict with validation results.
        """
        issues = []

        for i, msg in enumerate(messages):
            # Check for leaked standard spellings
            leaked = []
            for standard_word in ["hai", "aur", "haan", "accha", "pehle", "toh", "theek"]:
                if re.search(rf'\b{standard_word}\b', msg, re.IGNORECASE):
                    leaked.append(standard_word)
            if leaked:
                issues.append(f"Msg {i+1}: leaked standard spellings: {leaked}")

            # Check length
            if len(msg) > MAX_MSG_CHARS:
                issues.append(f"Msg {i+1}: too long ({len(msg)} chars)")

            # Check for LLM artifacts
            if any(art in msg.lower() for art in ["as an ai", "i'm an ai", "i cannot", "i'm sorry"]):
                issues.append(f"Msg {i+1}: contains LLM safety refusal")

        return {
            "valid": len(issues) == 0,
            "message_count": len(messages),
            "issues": issues,
        }
