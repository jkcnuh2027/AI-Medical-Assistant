import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
# Keep the new embedding space separate from vectors created by embedding-001.
NAMESPACE = "gemini-embedding-001-768-v1"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 768
