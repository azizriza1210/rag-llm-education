from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "machine_learning_modules"

TEMP_DIR = "data/documents"
TEMP_DIR.mkdir(exist_ok=True)
