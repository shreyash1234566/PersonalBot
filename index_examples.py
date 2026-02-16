"""
Index Builder
=============
Indexes the example bank into ChromaDB for semantic retrieval.
Run this once after the data pipeline, or whenever examples change.

Usage: python index_examples.py
"""

import time
from src.memory.vector_store import VectorStore


def main():
    print()
    print("█" * 55)
    print("  INDEX BUILDER — ChromaDB")
    print("█" * 55)
    print()

    start = time.time()

    store = VectorStore()
    count = store.index_example_bank()

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Collection: {store.collection_name}")
    print(f"  Documents: {count:,}")
    print()


if __name__ == "__main__":
    main()
