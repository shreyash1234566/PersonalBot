"""
Configuration
=============
Central config for API keys, model settings, paths, and tunable parameters.
API keys are loaded from .env file or environment variables.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"

PARSED_FILE = DATA_DIR / "parsed" / "parsed_messages.jsonl"
SESSIONS_FILE = DATA_DIR / "sessions" / "conversations.jsonl"
EXAMPLES_FILE = DATA_DIR / "examples" / "example_bank.jsonl"
STYLE_BIBLE_FILE = CONFIG_DIR / "style_bible.json"
PEOPLE_FILE = CONFIG_DIR / "people.json"

CHROMA_DIR = DATA_DIR / "chromadb"
HISTORY_DB = DATA_DIR / "history.sqlite3"

# ─── Load .env ────────────────────────────────────────────────────────────────

def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)

_load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# ─── Cloud / Remote Storage ────────────────────────────────────────────────

FIRESTORE_PROJECT_ID = os.environ.get("FIRESTORE_PROJECT_ID", "")
FIRESTORE_DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")
FIRESTORE_SERVICE_ACCOUNT_JSON = os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON", "")
FIRESTORE_CONVERSATIONS_COLLECTION = os.environ.get("FIRESTORE_CONVERSATIONS_COLLECTION", "conversations")
FIRESTORE_MESSAGES_COLLECTION = os.environ.get("FIRESTORE_MESSAGES_COLLECTION", "messages")

SHEETS_SPREADSHEET_ID = os.environ.get("SHEETS_SPREADSHEET_ID", "")
SHEETS_TAB_NAME = os.environ.get("SHEETS_TAB_NAME", "Sheet1")
SHEETS_SERVICE_ACCOUNT_JSON = os.environ.get("SHEETS_SERVICE_ACCOUNT_JSON", "")

AUTORESPONDER_SHARED_SECRET = os.environ.get("AUTORESPONDER_SHARED_SECRET", "")

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "")
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}

USE_FIRESTORE_HISTORY = _env_bool("USE_FIRESTORE_HISTORY", default=False)
ENABLE_SHEETS_LOG = _env_bool("ENABLE_SHEETS_LOG", default=False)

# History backend: "sqlite" (local), "firestore", or "mongo" (Render)
HISTORY_BACKEND = os.environ.get("HISTORY_BACKEND", "sqlite")
MONGODB_URI = os.environ.get("MONGODB_URI", "")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "chatbot")

# Backward compat: honour old USE_FIRESTORE_HISTORY flag
if USE_FIRESTORE_HISTORY and HISTORY_BACKEND == "sqlite":
    HISTORY_BACKEND = "firestore"

# ─── Model Settings ──────────────────────────────────────────────────────────

# Provider priority order (fallback chain)
LLM_PROVIDERS = ["groq", "google", "together"]

# Groq - Llama 3.3 70B (30 req/min free)
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_RPM = 30
GROQ_MAX_TOKENS = 512

# Google AI Studio - Gemini 2.0 Flash (15 RPM free)
GOOGLE_MODEL = "gemini-2.0-flash"
GOOGLE_RPM = 15
GOOGLE_MAX_TOKENS = 512

# Together AI - Qwen2.5-72B (fallback)
TOGETHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_RPM = 60
TOGETHER_MAX_TOKENS = 512

# ─── Generation Parameters ───────────────────────────────────────────────────

TEMPERATURE = 0.75          # Slightly creative but consistent
TOP_P = 0.9
FREQUENCY_PENALTY = 0.3     # Avoid repetition
PRESENCE_PENALTY = 0.1

# ─── Context Engine Settings ─────────────────────────────────────────────────

# How many similar examples to retrieve from ChromaDB
RETRIEVAL_TOP_K = 5

# How many recent conversation turns to include
HISTORY_WINDOW = 10

# ChromaDB collection name
CHROMA_COLLECTION = "ayush_examples"

# Embedding model for ChromaDB (runs on CPU, ~80MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Post-Processor Settings ─────────────────────────────────────────────────

# Max characters per single message in a burst
MAX_MSG_CHARS = 150

# Max messages in a single burst
MAX_BURST_SIZE = 6

# ─── Identity ────────────────────────────────────────────────────────────────

USER_IDENTITY = "I Am All"
REAL_NAME = "Shreyash"
