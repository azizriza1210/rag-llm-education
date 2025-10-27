import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Environment detection
IS_RENDER = os.getenv("RENDER", "").lower() in ("true", "1", "yes")
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Path configuration based on environment
if IS_RENDER or IS_PRODUCTION:
    # Production: Use /tmp (only writable directory on Render)
    TEMP_DIR = "/tmp/uploads"
    CHROMA_PERSIST_DIRECTORY = "/tmp/chroma_db"
    print("🚀 Running in PRODUCTION mode (Render)")
else:
    # Development: Use local directories
    TEMP_DIR = str(BASE_DIR / "data" / "documents")
    CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
    print("💻 Running in DEVELOPMENT mode (Local)")

# Collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "machine_learning_modules")

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Logging configuration
print("=" * 60)
print("📋 Configuration Loaded:")
print(f"  📁 BASE_DIR: {BASE_DIR}")
print(f"  📁 TEMP_DIR: {TEMP_DIR}")
print(f"  📁 CHROMA_PERSIST_DIRECTORY: {CHROMA_PERSIST_DIRECTORY}")
print(f"  📦 COLLECTION_NAME: {COLLECTION_NAME}")
print(f"  🔑 GROQ_API_KEY: {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")
print(f"  🔑 HUGGINGFACE_API_KEY: {'✅ Set' if HUGGINGFACE_API_KEY else '❌ Not set'}")
print(f"  🌍 Environment: {'Production (Render)' if IS_RENDER else 'Development (Local)'}")
print("=" * 60)

# Validate critical configs
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set! LLM features will not work.")

if not (HUGGINGFACE_API_KEY):
    print("⚠️  WARNING: No embeddings API key set! Please set COHERE_API_KEY or HUGGINGFACE_API_KEY.")