from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import NAMESPACE
from logger import logger
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.vectorstore import get_embeddings, get_index

router = APIRouter()


class SimpleRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.documents


@router.post("/ask/")
def ask_question(question: str = Form(...)):
    question = question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be blank")
    try:
        index = get_index()
        embedded_query = get_embeddings().embed_query(question)
        result = index.query(vector=embedded_query, top_k=3,
                             include_metadata=True, namespace=NAMESPACE)
        docs = []
        for match in result["matches"]:
            metadata = match.get("metadata") or {}
            if metadata.get("text"):
                docs.append(Document(page_content=metadata["text"], metadata=metadata))
        if not docs:
            return {"response": "No document text was found. Please upload PDFs first.",
                    "sources": []}
        chain = get_llm_chain(SimpleRetriever(documents=docs))
        return query_chain(chain, question)
    except Exception:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": "Unable to process question"})
