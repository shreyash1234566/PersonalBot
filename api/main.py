"""
Cloud API for AutoResponder
===========================
Exposes a /webhook endpoint that accepts JSON and returns a reply.
"""

import json
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from src.chatbot import Chatbot
from src.config import (
    AUTORESPONDER_SHARED_SECRET,
    HISTORY_BACKEND,
    PEOPLE_FILE,
)


def _load_people() -> dict:
    try:
        with open(PEOPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _resolve_partner_name(sender_name: str) -> str:
    people = _load_people()
    partners = people.get("partners", {})
    if not sender_name:
        return "a girl"
    needle = sender_name.strip().lower()
    for name, info in partners.items():
        aliases = [name] + info.get("aliases", [])
        if any(needle == a.lower() for a in aliases if isinstance(a, str)):
            return name
    return sender_name


def _parse_payload(payload: dict[str, Any]) -> dict[str, str]:
    data = payload
    if "query" in payload and isinstance(payload["query"], dict):
        data = payload["query"]
    elif "data" in payload and isinstance(payload["data"], dict):
        data = payload["data"]

    message = (
        data.get("message")
        or data.get("message_text")
        or data.get("text")
        or data.get("body")
        or ""
    )
    sender_name = (
        data.get("sender")
        or data.get("sender_name")
        or data.get("from_name")
        or data.get("chat_name")
        or data.get("name")
        or ""
    )
    group_participant = data.get("groupParticipant") or ""
    is_group = bool(data.get("isGroup"))
    sender_id = (
        data.get("sender_id")
        or data.get("from")
        or data.get("chat_id")
        or data.get("conversation_id")
        or data.get("phone")
        or ""
    )

    return {
        "message": str(message).strip(),
        "sender_id": str(sender_id).strip(),
        "sender_name": str(sender_name).strip(),
        "group_participant": str(group_participant).strip(),
        "is_group": "1" if is_group else "0",
    }


def _build_bot() -> Chatbot:
    if HISTORY_BACKEND == "firestore":
        from src.memory.firestore_history import FirestoreHistory
        history = FirestoreHistory()
    elif HISTORY_BACKEND == "mongo":
        from src.memory.mongo_history import MongoHistory
        history = MongoHistory()
    else:
        from src.memory.history import ConversationHistory
        history = ConversationHistory()
    return Chatbot(history=history)


BOT = _build_bot()


app = FastAPI(title="Shreyash WhatsApp Twin")


@app.get("/")
async def health():
    """Health check for Cloud Run."""
    return {"status": "ok", "bot": "Shreyash WhatsApp Twin"}


@app.post("/webhook")
async def webhook(request: Request):
    if AUTORESPONDER_SHARED_SECRET:
        token = request.headers.get("X-Auth-Token") or request.query_params.get("token")
        if token != AUTORESPONDER_SHARED_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    parsed = _parse_payload(payload)

    if not parsed["message"]:
        raise HTTPException(status_code=400, detail="Missing message text")

    sender_name = parsed["sender_name"]
    participant = parsed.get("group_participant", "")
    is_group = parsed.get("is_group") == "1"

    if is_group and participant:
        partner_name = _resolve_partner_name(participant)
        conversation_id = parsed["sender_id"] or f"group:{sender_name}:{participant}" or "default"
    else:
        partner_name = _resolve_partner_name(sender_name)
        conversation_id = parsed["sender_id"] or partner_name or "default"

    try:
        responses = await BOT.respond(
            parsed["message"],
            conversation_id=conversation_id,
            partner_name=partner_name,
        )
    except Exception as e:
        traceback.print_exc()
        return {"replies": [{"message": "Hmm"}]}

    return {"replies": [{"message": msg} for msg in responses]}


# ── Feedback (Rate Responses) ────────────────────────────────────────────

@app.post("/rate")
async def rate_response(request: Request):
    """
    Rate a bot response as good/bad for future improvement.
    Logs to a JSONL file for later analysis.

    Payload: {"conversation_id": "...", "rating": "good"|"bad",
             "message": "...", "response": "...", "note": "..."}
    """
    from datetime import datetime
    from pathlib import Path

    payload = await request.json()
    rating = payload.get("rating", "")
    if rating not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="rating must be 'good' or 'bad'")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_id": payload.get("conversation_id", ""),
        "rating": rating,
        "message": payload.get("message", ""),
        "response": payload.get("response", ""),
        "note": payload.get("note", ""),
    }

    feedback_file = Path("data/feedback.jsonl")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"saved": True, "rating": rating}


# ── Monitoring (Key Rotation Stats) ──────────────────────────────────────

@app.get("/stats")
async def stats():
    """
    Return Groq key rotation stats + overall bot health.
    Use this to monitor rate-limit behaviour in production.
    """
    groq_stats = {}
    if hasattr(BOT, "llm") and hasattr(BOT.llm, "_clients"):
        groq = BOT.llm._clients.get("groq")
        if groq and hasattr(groq, "get_stats"):
            groq_stats = groq.get_stats()

    return {
        "status": "ok",
        "history_backend": HISTORY_BACKEND,
        "groq": groq_stats,
    }
