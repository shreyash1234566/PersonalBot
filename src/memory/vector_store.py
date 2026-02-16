"""
Vector Store (ChromaDB)
=======================
Wraps ChromaDB for semantic retrieval of similar conversation examples.
Indexes the example bank and retrieves the most relevant examples
given a girl's message, for few-shot prompting.

Uses sentence-transformers all-MiniLM-L6-v2 (~80MB, CPU-only).
"""

import json
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EXAMPLES_FILE,
    RETRIEVAL_TOP_K,
)


class VectorStore:
    """ChromaDB-backed semantic retrieval for conversation examples."""

    def __init__(self, persist_dir: Path = None, collection_name: str = None):
        self.persist_dir = str(persist_dir or CHROMA_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    # ── Collection Management ─────────────────────────────────────────────

    @property
    def collection(self):
        """Lazy-load or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset(self):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_example_bank(self, filepath: Path = None, batch_size: int = 200):
        """
        Load example_bank.jsonl and index all examples into ChromaDB.

        Each document = the girl's message (context) for semantic matching.
        Metadata stores Ayush's response, categories, chat_id, etc.
        """
        src = filepath or EXAMPLES_FILE
        if not src.exists():
            raise FileNotFoundError(f"Example bank not found: {src}")

        # Reset collection for fresh indexing
        self.reset()

        examples = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        print(f"  Indexing {len(examples):,} examples into ChromaDB...")

        # Process in batches (ChromaDB has batch limits)
        documents = []
        metadatas = []
        ids = []

        for i, ex in enumerate(examples):
            # The document is the girl's message — this is what we search against
            context = ex["context"]
            if not context.strip():
                continue

            # Unique ID based on content hash
            content_hash = hashlib.md5(
                f"{ex['timestamp']}:{context}:{ex['response']}".encode()
            ).hexdigest()
            doc_id = f"ex_{content_hash[:12]}_{i}"

            # Truncate long preceding_context for metadata storage
            preceding = ex.get("preceding_context", [])
            preceding_str = json.dumps(preceding[-5:], ensure_ascii=False)
            if len(preceding_str) > 2000:
                preceding_str = preceding_str[:2000]

            documents.append(context)
            metadatas.append({
                "response": ex["response"],
                "categories": ",".join(ex["categories"]),
                "chat_id": ex["chat_id"],
                "timestamp": ex["timestamp"],
                "context_length": ex.get("context_length", 1),
                "preceding_context": preceding_str,
            })
            ids.append(doc_id)

        # Batch insert
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

        print(f"  ✓ Indexed {total:,} examples into '{self.collection_name}'")
        return total

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        chat_id: str = None,
        category: str = None,
    ) -> list[dict]:
        """
        Retrieve the most similar conversation examples for a query.

        Args:
            query: The girl's latest message(s) to match against.
            top_k: Number of results to return.
            chat_id: Optional filter by chat partner ("class_cr" or "shubhi").
            category: Optional filter by category.

        Returns:
            List of dicts with keys: context, response, categories, chat_id,
            distance, preceding_context.
        """
        k = top_k or RETRIEVAL_TOP_K
        if self.collection.count() == 0:
            return []

        # Build where filter
        where = None
        where_conditions = []

        if chat_id:
            where_conditions.append({"chat_id": chat_id})
        if category:
            where_conditions.append({"categories": {"$contains": category}})

        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count()),
                where=where,
            )
        except Exception as e:
            # Fallback: query without filters
            print(f"  [VectorStore] Filter query failed ({e}), retrying without filters")
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count()),
            )

        # Parse results
        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if results.get("distances") else [0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Parse preceding_context back from JSON string
                preceding = []
                if meta.get("preceding_context"):
                    try:
                        preceding = json.loads(meta["preceding_context"])
                    except (json.JSONDecodeError, TypeError):
                        preceding = []

                retrieved.append({
                    "context": doc,
                    "response": meta["response"],
                    "categories": meta.get("categories", "").split(","),
                    "chat_id": meta.get("chat_id", ""),
                    "distance": round(dist, 4),
                    "preceding_context": preceding,
                })

        return retrieved

    # ── Info ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return number of indexed documents."""
        return self.collection.count()

    def info(self) -> dict:
        """Return collection metadata."""
        return {
            "collection": self.collection_name,
            "count": self.count(),
            "persist_dir": self.persist_dir,
        }
