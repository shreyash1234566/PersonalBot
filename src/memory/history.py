"""
Conversation History (SQLite)
=============================
Stores ongoing conversation history with each girl for context continuity.
Tracks messages, timestamps, and session metadata.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.config import HISTORY_DB, HISTORY_WINDOW


class ConversationHistory:
    """SQLite-backed conversation history for session continuity."""

    def __init__(self, db_path: Path = None):
        self.db_path = str(db_path or HISTORY_DB)
        self._conn = None
        self._init_db()

    # ── Database Setup ────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                partner_name TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_active
                ON conversations(last_active DESC);
        """)
        self.conn.commit()

    # ── Conversation Management ───────────────────────────────────────────

    def get_or_create_conversation(
        self,
        conversation_id: str,
        partner_name: str = "Unknown",
    ) -> dict:
        """Get or create a conversation record."""
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if row:
            return dict(row)

        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO conversations
               (conversation_id, partner_name, created_at, last_active, message_count)
               VALUES (?, ?, ?, ?, 0)""",
            (conversation_id, partner_name, now, now),
        )
        self.conn.commit()
        return {
            "conversation_id": conversation_id,
            "partner_name": partner_name,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }

    def list_conversations(self) -> list[dict]:
        """List all conversations, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM conversations ORDER BY last_active DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Message Storage ───────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: str = None,
        metadata: dict = None,
    ):
        """Add a message to conversation history."""
        ts = timestamp or datetime.now().isoformat()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)

        self.conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content, ts, meta_str),
        )
        self.conn.execute(
            """UPDATE conversations
               SET last_active = ?, message_count = message_count + 1
               WHERE conversation_id = ?""",
            (ts, conversation_id),
        )
        self.conn.commit()

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = None,
    ) -> list[dict]:
        """
        Get the most recent messages from a conversation.
        Returns in chronological order (oldest first).
        """
        n = limit or HISTORY_WINDOW
        rows = self.conn.execute(
            """SELECT role, content, timestamp, metadata
               FROM messages
               WHERE conversation_id = ?
               ORDER BY timestamp DESC, id DESC
               LIMIT ?""",
            (conversation_id, n),
        ).fetchall()

        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            meta = {}
            try:
                meta = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": meta,
            })
        return messages

    def get_recent_as_chatml(
        self,
        conversation_id: str,
        limit: int = None,
    ) -> list[dict]:
        """Get recent messages in ChatML format (role + content only)."""
        messages = self.get_recent_messages(conversation_id, limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # ── Session Detection ─────────────────────────────────────────────────

    def is_new_session(
        self,
        conversation_id: str,
        gap_hours: float = 2.0,
    ) -> bool:
        """Check if enough time has passed to consider this a new session."""
        row = self.conn.execute(
            """SELECT timestamp FROM messages
               WHERE conversation_id = ?
               ORDER BY timestamp DESC, id DESC
               LIMIT 1""",
            (conversation_id,),
        ).fetchone()

        if not row:
            return True

        last_time = datetime.fromisoformat(row["timestamp"])
        now = datetime.now()
        return (now - last_time) > timedelta(hours=gap_hours)

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self, conversation_id: str = None) -> dict:
        """Get message count stats."""
        if conversation_id:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return {"conversation_id": conversation_id, "message_count": row["cnt"]}
        
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM messages").fetchone()
        conv_row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM conversations"
        ).fetchone()
        return {
            "total_messages": row["cnt"],
            "total_conversations": conv_row["cnt"],
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear_conversation(self, conversation_id: str):
        """Delete all messages and the conversation record."""
        self.conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        self.conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        self.conn.commit()

    def clear_all(self):
        """Delete everything."""
        self.conn.execute("DELETE FROM messages")
        self.conn.execute("DELETE FROM conversations")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
