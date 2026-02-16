"""
Conversation History (Firestore)
===============================
Cloud-backed conversation history for context continuity.
"""

import json
from datetime import datetime, timedelta

from google.cloud import firestore
from google.oauth2 import service_account

from src.config import (
    FIRESTORE_PROJECT_ID,
    FIRESTORE_DATABASE,
    FIRESTORE_SERVICE_ACCOUNT_JSON,
    FIRESTORE_CONVERSATIONS_COLLECTION,
    FIRESTORE_MESSAGES_COLLECTION,
    HISTORY_WINDOW,
)


class FirestoreHistory:
    """Firestore-backed conversation history for session continuity."""

    def __init__(self):
        self._client = None

    @property
    def client(self) -> firestore.Client:
        if self._client is None:
            credentials = None
            if FIRESTORE_SERVICE_ACCOUNT_JSON:
                info = json.loads(FIRESTORE_SERVICE_ACCOUNT_JSON)
                credentials = service_account.Credentials.from_service_account_info(info)
            if FIRESTORE_PROJECT_ID:
                self._client = firestore.Client(
                    project=FIRESTORE_PROJECT_ID,
                    database=FIRESTORE_DATABASE,
                    credentials=credentials,
                )
            else:
                self._client = firestore.Client(
                    database=FIRESTORE_DATABASE,
                    credentials=credentials,
                )
        return self._client

    # ── Conversation Management ───────────────────────────────────────────

    def get_or_create_conversation(self, conversation_id: str, partner_name: str = "Unknown") -> dict:
        doc_ref = self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION).document(conversation_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()

        now = datetime.utcnow()
        data = {
            "conversation_id": conversation_id,
            "partner_name": partner_name,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        doc_ref.set(data)
        return data

    def list_conversations(self) -> list[dict]:
        docs = (
            self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION)
            .order_by("last_active", direction=firestore.Query.DESCENDING)
            .stream()
        )
        return [d.to_dict() for d in docs]

    # ── Message Storage ───────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str = None,
        metadata: dict = None,
    ):
        ts = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
        meta = metadata or {}

        msg = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": ts,
            "metadata": meta,
        }
        self.client.collection(FIRESTORE_MESSAGES_COLLECTION).add(msg)
        self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION).document(conversation_id).set(
            {
                "last_active": ts,
                "message_count": firestore.Increment(1),
            },
            merge=True,
        )

    def get_recent_messages(self, conversation_id: str, limit: int = None) -> list[dict]:
        n = limit or HISTORY_WINDOW
        docs = (
            self.client.collection(FIRESTORE_MESSAGES_COLLECTION)
            .where("conversation_id", "==", conversation_id)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(n)
            .stream()
        )

        rows = list(docs)
        messages = []
        for doc in reversed(rows):
            data = doc.to_dict()
            ts = data.get("timestamp")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            messages.append(
                {
                    "role": data.get("role"),
                    "content": data.get("content"),
                    "timestamp": ts_str,
                    "metadata": data.get("metadata") or {},
                }
            )
        return messages

    def get_recent_as_chatml(self, conversation_id: str, limit: int = None) -> list[dict]:
        messages = self.get_recent_messages(conversation_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # ── Session Detection ─────────────────────────────────────────────────

    def is_new_session(self, conversation_id: str, gap_hours: float = 2.0) -> bool:
        docs = (
            self.client.collection(FIRESTORE_MESSAGES_COLLECTION)
            .where("conversation_id", "==", conversation_id)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        last = next(docs, None)
        if not last:
            return True
        last_ts = last.to_dict().get("timestamp")
        if not last_ts:
            return True
        now = datetime.utcnow()
        return (now - last_ts) > timedelta(hours=gap_hours)

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self, conversation_id: str = None) -> dict:
        if conversation_id:
            docs = (
                self.client.collection(FIRESTORE_MESSAGES_COLLECTION)
                .where("conversation_id", "==", conversation_id)
                .stream()
            )
            count = sum(1 for _ in docs)
            return {"conversation_id": conversation_id, "message_count": count}

        conv_docs = self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION).stream()
        msg_docs = self.client.collection(FIRESTORE_MESSAGES_COLLECTION).stream()
        return {
            "total_conversations": sum(1 for _ in conv_docs),
            "total_messages": sum(1 for _ in msg_docs),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear_conversation(self, conversation_id: str):
        conv_ref = self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION).document(conversation_id)
        conv_ref.delete()
        docs = (
            self.client.collection(FIRESTORE_MESSAGES_COLLECTION)
            .where("conversation_id", "==", conversation_id)
            .stream()
        )
        for doc in docs:
            doc.reference.delete()

    def clear_all(self):
        for doc in self.client.collection(FIRESTORE_MESSAGES_COLLECTION).stream():
            doc.reference.delete()
        for doc in self.client.collection(FIRESTORE_CONVERSATIONS_COLLECTION).stream():
            doc.reference.delete()

    def close(self):
        self._client = None
