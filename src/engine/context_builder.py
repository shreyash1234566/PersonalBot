"""
Context Builder
===============
Assembles the full prompt for the LLM by combining:
  1. System prompt (style bible rules)
  2. Retrieved similar examples (few-shot)
  3. Conversation history (recent turns)
  4. The girl's latest message

This is the brain that decides exactly what context the LLM sees.
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import (
    STYLE_BIBLE_FILE,
    RETRIEVAL_TOP_K,
    HISTORY_WINDOW,
    PEOPLE_FILE,
)


class ContextBuilder:
    """Builds LLM prompts from style bible + examples + history."""

    def __init__(self, style_bible_path: Path = None, people_path: Path = None):
        self._bible = None
        self._bible_path = style_bible_path or STYLE_BIBLE_FILE
        self._people = None
        self._people_path = people_path or PEOPLE_FILE

    # ── Style Bible ───────────────────────────────────────────────────────

    @property
    def style_bible(self) -> dict:
        """Lazy-load the style bible."""
        if self._bible is None:
            with open(self._bible_path, "r", encoding="utf-8") as f:
                self._bible = json.load(f)
        return self._bible

    @property
    def people_data(self) -> dict:
        """Lazy-load partner profiles."""
        if self._people is None:
            try:
                with open(self._people_path, "r", encoding="utf-8") as f:
                    self._people = json.load(f)
            except FileNotFoundError:
                self._people = {}
        return self._people

    def _find_partner_profile(self, partner_name: str) -> tuple[str, dict] | None:
        if not partner_name:
            return None
        people = self.people_data or {}
        partners = people.get("partners", {})
        needle = partner_name.strip().lower()
        for name, info in partners.items():
            aliases = [name] + info.get("aliases", [])
            if any(needle == a.lower() for a in aliases if isinstance(a, str)):
                return name, info
        return None

    def _render_personal_context(self, partner_name: str) -> str:
        people = self.people_data or {}
        self_info = people.get("self", {})
        profile = self._find_partner_profile(partner_name)
        if not self_info and not profile:
            return ""

        lines = ["═══ PERSONAL CONTEXT ═══"]
        if self_info:
            lines.append("You are Shreyash:")
            if self_info.get("study"):
                lines.append(f"• Study: {self_info['study']}")
            traits = self_info.get("traits", [])
            if traits:
                lines.append("• Traits: " + ", ".join(traits))
            tone_notes = self_info.get("tone_notes", [])
            if tone_notes:
                lines.append("• Tone: " + "; ".join(tone_notes))

        if profile:
            name, info = profile
            lines.append(f"Talking to: {name}")
            if info.get("relationship"):
                lines.append(f"• Relationship: {info['relationship']}")
            if info.get("current"):
                lines.append(f"• Current: {info['current']}")
            if info.get("age"):
                lines.append(f"• Age: {info['age']}")
            if info.get("birthday"):
                lines.append(f"• Birthday: {info['birthday']}")
            if info.get("family"):
                lines.append(f"• Family: {info['family']}")
            if info.get("interests"):
                lines.append("• Interests: " + ", ".join(info["interests"]))
            if info.get("personality"):
                lines.append("• Personality: " + ", ".join(info["personality"]))
            if info.get("behavior"):
                lines.append("• Behavior: " + ", ".join(info["behavior"]))
            if info.get("nicknames"):
                lines.append("• Nicknames you use for her: " + ", ".join(info["nicknames"]))
            if info.get("inside_jokes"):
                lines.append("• Inside jokes between you two: " + "; ".join(info["inside_jokes"]))
            if info.get("friends"):
                lines.append("• Mutual/her friends: " + ", ".join(info["friends"]))
            if info.get("history"):
                lines.append(f"• Background: {info['history']}")
            if info.get("conversation_notes"):
                lines.append("• Notes: " + "; ".join(info["conversation_notes"]))
            if info.get("tone_preference"):
                lines.append("• Tone rules: " + "; ".join(info["tone_preference"]))
            if info.get("do_not_mention"):
                lines.append("• Do NOT mention: " + ", ".join(info["do_not_mention"]))

        return "\n".join(lines)

    # ── System Prompt ─────────────────────────────────────────────────────

    def build_system_prompt(self, partner_name: str = "a girl") -> str:
        """
        Build a comprehensive system prompt from the style bible.
        Encodes ALL of Shreyash's texting rules so the LLM can replicate them.
        """
        bible = self.style_bible

        # Extract top spelling rules
        spelling_rules = []
        for word, info in bible.get("spelling_map", {}).items():
            if info.get("confirmed") and info.get("ayush_count", 0) > 10:
                spelling_rules.append(
                    f'"{word}" NOT "{info["standard_form"]}"'
                )

        # Extract greetings
        morning = bible.get("greetings_and_closings", {}).get("morning_greetings", [])
        night = bible.get("greetings_and_closings", {}).get("night_closings", [])
        morning_top = morning[0]["text"] if morning else "Good morning 🌄🌄🌄"
        night_top = night[0]["text"] if night else "Good night 🌉🌉🌉"

        # Extract short responses
        short_resp = bible.get("short_responses", {}).get("top_short_responses", [])
        short_list = [r["text"] for r in short_resp[:8]]

        # Message length stats
        lengths = bible.get("message_lengths", {})
        avg_chars = lengths.get("char_length", {}).get("mean", 24)
        median_chars = lengths.get("char_length", {}).get("median", 16)
        avg_words = lengths.get("word_count", {}).get("mean", 5.5)

        # Burst patterns
        bursts = bible.get("burst_patterns", {})
        avg_burst = bursts.get("avg_burst_length", 1.6)
        multi_pct = bursts.get("multi_message_pct", 39)

        # Time + day awareness
        now = datetime.now()
        hour = now.hour
        day_name = now.strftime("%A")
        weekday = now.weekday()  # 0=Mon, 6=Sun

        if 5 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 17:
            time_context = "afternoon"
        elif 17 <= hour < 21:
            time_context = "evening"
        else:
            time_context = "late night"

        # Shreyash's college schedule:
        # Mon, Tue, Wed, Fri = college 8 AM - 2 PM
        # Thu, Sat, Sun = holiday
        college_days = {0, 1, 2, 4}  # Mon, Tue, Wed, Fri
        is_college_day = weekday in college_days
        in_class = is_college_day and 8 <= hour < 14

        if in_class:
            activity_context = "in college class right now (might reply late)"
        elif is_college_day and hour < 8:
            activity_context = "getting ready for college"
        elif is_college_day and 14 <= hour < 16:
            activity_context = "just got back from college, in PG"
        elif is_college_day:
            activity_context = "at PG, free after college"
        elif weekday == 3:
            activity_context = "holiday today (no college on Thursday), chilling at PG"
        elif weekday == 5:
            activity_context = "weekend, no college today, at PG"
        else:
            activity_context = "Sunday, full holiday, at PG"

        system = f"""You are Shreyash ("I Am All"), a Hinglish-speaking Indian guy chatting with {partner_name}. You MUST perfectly replicate Shreyash's exact texting style. Your responses should be INDISTINGUISHABLE from the real Shreyash.

═══ SPELLING RULES (MANDATORY — NEVER VIOLATE) ═══
Use EXACTLY these spellings. NEVER use the standard form:
{chr(10).join(f'• {rule}' for rule in spelling_rules[:20])}

═══ MESSAGE FORMAT ═══
• Keep messages SHORT: ~{avg_chars:.0f} chars, ~{avg_words:.0f} words (median {median_chars} chars)
• {multi_pct:.0f}% of your replies are multi-message bursts (avg {avg_burst:.1f} msgs)
• Use [MSG_BREAK] between separate messages in a burst
• Example burst: "Ha sahi h [MSG_BREAK] Me bhi esa hi sochta tha [MSG_BREAK] Ab chod"
• Single-word responses are common: {', '.join(f'"{s}"' for s in short_list[:6])}
• NEVER write long paragraphs. Break thoughts into rapid-fire short messages.

═══ EMOJI RULES (CRITICAL — READ CAREFULLY) ═══
• 82% of your messages have ZERO emoji. The DEFAULT is plain text with NO emoji.
• Only ~18% of messages contain any emoji at all.
• When you DO use emoji (rarely), use at most ONE per message.
• NEVER put emoji in every message. A typical 5-message burst has 0-1 emoji total.
• Allowed emojis (use sparingly): 😅 😈 🤣 🤔 👆
• 🌄 = morning greeting ONLY. 🌉 = night greeting ONLY.
• If unsure whether to add emoji → DON'T. Plain text is almost always correct.

═══ CASUAL RESPONSE PATTERNS (USE THESE) ═══
• "Kkrh" = "kya kar raha/rahi" — use this exact abbreviation
• "Tp" = "timepass" — Use when the topic is about what you're doing and you're free. Can also be used if she says "me bhi kuch nahi" in a kkrh thread. But NEVER use "Tp" as a random reply to unrelated messages.
• When asked "kkrh?" or "what are you doing?", reply based on your ACTUAL schedule/activity (see GREETINGS & SCHEDULE below). Use varied answers like "Class me hu", "PG pe hu bs", "Kyuch nahi", "Bs phone chala rha", "Tp" — pick what fits the current time.
• "Sahi h" = casual agreement with a STATEMENT, NOT an answer to a question. If she asks "did you do X?", answer with "Ha" or "Nhi" — NOT "Sahi h".
• "Aacha" = understanding/acknowledgement  
• "Ha to" = "yeah so" — casual filler
• "Bol" / "Bta" = "tell me" — casual prompt
• "Chl" / "Chl bye" = casual goodbye
• "Oook" = drawn-out ok (NOT "Ok" or "Okay")
• MIRRORING: If she says "hmm", reply with "mm". If she says "mm", reply with "hmm". This is a signature pattern.

═══ CRITICAL: WHEN NOT TO USE CASUAL FILLERS ═══
• NEVER reply with "Tp", "Kkrh", "Hmm" or other filler when she asks a real question or shares something meaningful.
• If she tells you something emotional, sad, or important — RESPOND to the content, don't deflect with fillers.
• If you don't understand her message, ask "Kya?" or "Mtlb?" — don't reply with random "Tp" or "Hmm".
• "Tp" is for when the conversation is about activities/free time. "Hmm" is ONLY for low-energy acknowledgement of simple statements.
• If she sends a message you genuinely can't parse (gibberish, sticker-only), it's ok to say "Kya hua?" or just ignore.
• AVOID repeating "Sahi h" at the start of every reply — vary your acknowledgements: "Ha", "Aacha", "Ha to", "Hmm".

═══ GREETINGS & SCHEDULE ═══
• Morning: "{morning_top}" (exact format, with emojis)
• Night: "{night_top}" (exact format, with emojis)
• Current time: {time_context} ({now.strftime("%I:%M %p")}), {day_name}
• Today: {"College day (Mon/Tue/Wed/Fri 8AM-2PM)" if is_college_day else "Holiday (no college)"}
• Right now: {activity_context}
• When asked "kkrh?" or "kya kar rha?", answer based on your ACTUAL current activity above. Vary your answers — don't always say the same thing.

═══ PERSONALITY & TONE ═══
• Casually cool. Chill. Don't try too hard.
• Witty, slightly sarcastic humor. Teasing but caring.
• NEVER formal or polite ("ji", "aapka", "kripya" are BANNED)
• NEVER overtly romantic or cheesy. Keep it real.
• Uses "tu/tera/teri" (informal) with close people
• Deflects serious topics with light humor, but shows genuine care when it matters
• References: BGMI, coding, college, friends, inside jokes
• NEVER uses English punctuation excessively. No "!!!" or "???". Minimal commas.
• Capitalize first word only. Rest lowercase.

═══ RESPONSE STRATEGY ═══
• Mirror the girl's energy — playful→more playful, serious→humor+care
• If she asks a question, answer directly then add a follow-up QUESTION to keep the conversation going
• If she states an opinion ("X best h"), DON'T just agree. Either share your own take, tease her, or ask a follow-up. Avoid generic agreement like "Sahi h | Mera bhi same h"
• If she shares something emotional, acknowledge briefly then lighten mood
• If she sends "Hmm"/"Ok" type messages, either tease or change topic
• NEVER be dry. NEVER just say "ok" back.
• If you don't know what to say, tease her or ask something about her day
• When she mentions people/things you know about (from PERSONAL CONTEXT), reference them naturally — but don't force inside jokes into every message

═══ CRITICAL: ACTUALLY UNDERSTAND THE CONVERSATION ═══
• READ the full conversation history carefully before responding.
• Your response MUST make logical sense given what she just said.
• If she asks "matalb?" or "what do you mean?" → EXPLAIN what you meant, don't deflect.
• If she accuses you of something or questions you → RESPOND to her actual point.
• If she says something doesn't make sense → ACKNOWLEDGE and clarify.
• NEVER give random pattern-matched responses that ignore her message.
• "Kkrh" and "Tp" are ONLY valid when she asks what you're doing, NOT as random filler.
• Each response must DIRECTLY address what she just said."""

        personal_context = self._render_personal_context(partner_name)
        if personal_context:
            system += "\n\n" + personal_context

        return system

    # ── Few-Shot Examples ─────────────────────────────────────────────────

    def format_examples(self, retrieved: list[dict], max_examples: int = None) -> str:
        """
        Format retrieved examples as few-shot demonstrations.
        
        Args:
            retrieved: List from VectorStore.retrieve()
            max_examples: Cap on number of examples to include
            
        Returns:
            Formatted string for the prompt
        """
        if not retrieved:
            return ""

        n = max_examples or RETRIEVAL_TOP_K
        examples = retrieved[:n]

        lines = ["═══ REFERENCE EXAMPLES (Shreyash's real replies in similar situations) ═══"]
        lines.append("Study these and match the EXACT style, length, and tone:\n")

        for i, ex in enumerate(examples, 1):
            ctx = ex["context"].replace("[MSG_BREAK]", " → ")
            resp = ex["response"].replace("[MSG_BREAK]", " → ")
            cats = ", ".join(ex.get("categories", []))

            lines.append(f"Example {i} [{cats}]:")
            lines.append(f"  Her: {ctx}")
            lines.append(f"  Shreyash: {resp}")
            lines.append("")

        return "\n".join(lines)

    # ── Full Prompt Assembly ──────────────────────────────────────────────

    def build_messages(
        self,
        girl_message: str,
        partner_name: str = "a girl",
        history: list[dict] = None,
        retrieved_examples: list[dict] = None,
    ) -> list[dict]:
        """
        Assemble the complete ChatML message list for the LLM.

        Structure:
          1. [system] Style bible rules + few-shot examples
          2. [user/assistant...] Conversation history
          3. [user] The girl's latest message

        Args:
            girl_message: The girl's latest message to respond to
            partner_name: Name of the chat partner for system prompt
            history: Recent ChatML turns from ConversationHistory
            retrieved_examples: Similar examples from VectorStore

        Returns:
            List of ChatML messages ready for LLM
        """
        messages = []

        # 1. System prompt
        system_text = self.build_system_prompt(partner_name)

        # 2. Attach few-shot examples to system prompt
        if retrieved_examples:
            examples_text = self.format_examples(retrieved_examples)
            if examples_text:
                system_text += "\n\n" + examples_text

        messages.append({"role": "system", "content": system_text})

        # 3. Conversation history
        if history:
            # Limit to HISTORY_WINDOW turns
            recent = history[-HISTORY_WINDOW:]
            for turn in recent:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"],
                })

        # 4. Girl's latest message
        messages.append({"role": "user", "content": girl_message})

        return messages

    # ── Token Estimation ──────────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate (~4 chars per token for Hinglish)."""
        total_chars = sum(len(m["content"]) for m in messages)
        return total_chars // 4
