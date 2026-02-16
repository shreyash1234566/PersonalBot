"""
Run Pipeline
============
Master script to execute the full Week 1 data pipeline:
  1. Parse WhatsApp chats → parsed_messages.jsonl
  2. Build conversation sessions → conversations.jsonl
  3. Analyze style → style_bible.json
  4. Generate example bank → example_bank.jsonl
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.parser import run as run_parser
from src.session_builder import run as run_sessions
from src.style_analyzer import run as run_style
from src.example_bank import run as run_examples


def main():
    start = time.time()
    
    print("\n" + "█" * 60)
    print("  PERSONAL CHATBOT — WEEK 1 DATA PIPELINE")
    print("█" * 60)
    
    # Step 1: Parse
    parsed = run_parser()
    
    # Step 2: Sessions
    conversations = run_sessions()
    
    # Step 3: Style Bible
    style = run_style()
    
    # Step 4: Example Bank
    examples = run_examples()
    
    elapsed = time.time() - start
    
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  Outputs:")
    print(f"    data/parsed/parsed_messages.jsonl  ({len(parsed):,} messages)")
    print(f"    data/sessions/conversations.jsonl  ({len(conversations):,} training examples)")
    print(f"    config/style_bible.json")
    print(f"    data/examples/example_bank.jsonl   ({len(examples):,} examples)")
    print()


if __name__ == "__main__":
    main()
