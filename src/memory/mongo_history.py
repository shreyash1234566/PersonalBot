"""
Conversation History (MongoDB)
==============================
MongoDB Atlas-backed conversation history for Render deployment.
Free tier: 512 MB shared cluster — more than enough for chat history.

Implements the same duck-typed interface as ConversationHistory (SQLite)
and FirestoreHistory so Chatbot works with any backend via DI.
"""

from datetime import datetime, timedelta

from pymongo import MongoClient, DESCENDING

from src.config import MONGODB_URI, MONGODB_DATABASE, HISTORY_WINDOW


class MongoHistory:
    """MongoDB-backed conversation history with lazy connection."""

    def __init__(self, uri: str = None, database: str = None):
        self._uri = uri or MONGODB_URI
        self._db_name = database or MONGODB_DATABASE
        self._client: MongoClient | None = None
        self._db = None

    # ── Lazy Connection ───────────────────────────────────────────────

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            try:
                self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
                # attempt to connect and surface any connection errors early
                try:
                    info = self._client.server_info()
                    print(f"[mongo_history] connected to MongoDB server version={info.get('version')}")
                except Exception as conn_err:
                    print(f"[mongo_history] warning: could not retrieve server_info: {conn_err}")
            except Exception as e:
                print(f"[mongo_history] ERROR creating MongoClient: {e}")
                raise
        return self._client

    @property
    def db(self):
        if self._db is None:
            self._db = self.client[self._db_name]
            # Create indexes once on first access
            self._db["messages"].create_index(
                [("conversation_id", 1), ("timestamp", -1)]
            )
            self._db["conversations"].create_index("conversation_id", unique=True)
            self._db["conversations"].create_index([("last_active", -1)])
        return self._db

    # ── Conversation Management ───────────────────────────────────────

    def get_or_create_conversation(
        self, conversation_id: str, partner_name: str = "Unknown"
    ) -> dict:
        doc = self.db["conversations"].find_one(
            {"conversation_id": conversation_id}
        )
        if doc:
            doc.pop("_id", None)
            return doc

        now = datetime.utcnow().isoformat()
        data = {
            "conversation_id": conversation_id,
            "partner_name": partner_name,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self.db["conversations"].insert_one(data)
        data.pop("_id", None)
        return data

    def list_conversations(self) -> list[dict]:
        docs = self.db["conversations"].find().sort("last_active", DESCENDING)
        results = []
        for d in docs:
            d.pop("_id", None)
            results.append(d)
        return results

    # ── Message Storage ───────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str = None,
        metadata: dict = None,
    ):
        ts = timestamp or datetime.utcnow().isoformat()
        msg = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": ts,
            "metadata": metadata or {},
        }
        try:
            res = self.db["messages"].insert_one(msg)
            print(f"[mongo_history] insert_one ok id={res.inserted_id}")
        except Exception as e:
            print(f"[mongo_history] ERROR inserting message for {conversation_id}: {e}")
            import traceback

            traceback.print_exc()
            # do not re-raise so the bot can still reply; caller can inspect logs
            return

        try:
            self.db["conversations"].update_one(
                {"conversation_id": conversation_id},
                {"$set": {"last_active": ts}, "$inc": {"message_count": 1}},
            )
        except Exception as e:
            print(f"[mongo_history] ERROR updating conversation metadata for {conversation_id}: {e}")
            import traceback

            traceback.print_exc()

    def get_recent_messages(
        self, conversation_id: str, limit: int = None
    ) -> list[dict]:
        n = limit or HISTORY_WINDOW
        cursor = (
            self.db["messages"]
            .find({"conversation_id": conversation_id})
            .sort("timestamp", DESCENDING)
            .limit(n)
        )
        rows = list(cursor)
        # Reverse so oldest-first (same order as SQLite backend)
        messages = []
        for row in reversed(rows):
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row.get("timestamp", ""),
                    "metadata": row.get("metadata", {}),
                }
            )
        return messages

    def get_recent_as_chatml(
        self, conversation_id: str, limit: int = None
    ) -> list[dict]:
        messages = self.get_recent_messages(conversation_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # ── Session Detection ─────────────────────────────────────────────

    def is_new_session(
        self, conversation_id: str, gap_hours: float = 2.0
    ) -> bool:
        last = self.db["messages"].find_one(
            {"conversation_id": conversation_id},
            sort=[("timestamp", DESCENDING)],
        )
        if not last:
            return True
        try:
            last_ts = datetime.fromisoformat(last["timestamp"])
        except (ValueError, KeyError):
            return True
        return (datetime.utcnow() - last_ts) > timedelta(hours=gap_hours)

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self, conversation_id: str = None) -> dict:
        if conversation_id:
            count = self.db["messages"].count_documents(
                {"conversation_id": conversation_id}
            )
            return {"conversation_id": conversation_id, "message_count": count}
        return {
            "total_messages": self.db["messages"].count_documents({}),
            "total_conversations": self.db["conversations"].count_documents({}),
        }

    # ── Cleanup ───────────────────────────────────────────────────────

    def clear_conversation(self, conversation_id: str):
        self.db["messages"].delete_many({"conversation_id": conversation_id})
        self.db["conversations"].delete_one(
            {"conversation_id": conversation_id}
        )

    def clear_all(self):
        self.db["messages"].delete_many({})
        self.db["conversations"].delete_many({})

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
