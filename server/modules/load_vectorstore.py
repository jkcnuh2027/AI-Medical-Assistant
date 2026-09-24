import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import NAMESPACE
from modules.vectorstore import get_embeddings, get_index

MAX_FILE_BYTES = 20 * 1024 * 1024


def load_vectorstore(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    with TemporaryDirectory() as directory:
        for number, file in enumerate(uploaded_files):
            name = Path(file.filename or "document.pdf").name
            if not name.lower().endswith(".pdf"):
                raise HTTPException(400, "Only PDF files are supported")
            content = file.file.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, "Each PDF must be at most 20 MB")
            if not content.startswith(b"%PDF-"):
                raise HTTPException(400, "Invalid PDF file")
            path = Path(directory) / f"{number}.pdf"
            path.write_bytes(content)
            try:
                documents = PyPDFLoader(str(path)).load()
            except Exception as exc:
                raise HTTPException(400, "Unable to read PDF") from exc
            chunks = splitter.split_documents(documents)
            if not chunks:
                raise HTTPException(400, "PDF has no extractable text; scanned PDFs need OCR")
            digest = hashlib.sha256(content).hexdigest()
            for i, chunk in enumerate(chunks):
                # Pinecone metadata accepts scalar values, not arbitrary PDF metadata.
                metadata = {"text": chunk.page_content, "source": name,
                            "page": chunk.metadata.get("page", 0)}
                all_chunks.append((f"{digest}-{i}", chunk.page_content, metadata))

    if not all_chunks:
        raise HTTPException(400, "Upload at least one PDF")
    embeddings = get_embeddings()
    index = get_index(create=True)
    for start in range(0, len(all_chunks), 64):
        batch = all_chunks[start:start + 64]
        vectors = embeddings.embed_documents([item[1] for item in batch])
        index.upsert(vectors=[
            {"id": item[0], "values": vector, "metadata": item[2]}
            for item, vector in zip(batch, vectors, strict=True)
        ], namespace=NAMESPACE)
