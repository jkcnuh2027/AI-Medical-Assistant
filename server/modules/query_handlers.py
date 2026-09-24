def query_chain(chain, user_input: str):
    result = chain.invoke({"query": user_input})
    return {
        "response": result["result"],
        "sources": list(dict.fromkeys(
            doc.metadata["source"] for doc in result["source_documents"]
            if doc.metadata.get("source")
        )),
    }
