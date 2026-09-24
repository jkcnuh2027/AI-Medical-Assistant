import os
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, INDEX_NAME, require_env


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=require_env("GOOGLE_API_KEY"),
        output_dimensionality=EMBEDDING_DIMENSION,
    )


def get_index(create=False):
    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    if create and INDEX_NAME not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")
                ),
            )
        except Exception as exc:
            if getattr(exc, "status", None) != 409:
                raise
    deadline = time.monotonic() + 60
    while True:
        description = pc.describe_index(INDEX_NAME)
        if description.dimension != EMBEDDING_DIMENSION:
            raise RuntimeError("Pinecone index must have 768 dimensions")
        if description.status["ready"]:
            return pc.Index(host=description.host)
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinecone index was not ready within 60 seconds")
        time.sleep(1)
