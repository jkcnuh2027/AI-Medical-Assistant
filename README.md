# AI Medical Assistant

FastAPI and Streamlit application for answering questions from uploaded PDFs using
Google embeddings, Pinecone retrieval, and Groq.

## Deploy the backend on Render

- Root Directory: `server`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Python: `.python-version` selects Python 3.12. Remove any old `PYTHON_VERSION`
  override in Render, or set it to a supported fully qualified 3.12 version.
- Add `GOOGLE_API_KEY`, `GROQ_API_KEY`, and `PINECONE_API_KEY` under Render Environment.
  Set `PINECONE_INDEX_NAME` if your index is not named `medicalindex`.
  See `server/.env.example` for optional settings. Local `.env` files are not deployed.

Deploy the updated commit with a cleared build cache. If your Render service has
no Root Directory, use `pip install -r server/requirements.txt` and
`uvicorn main:app --app-dir server --host 0.0.0.0 --port $PORT` instead.

`GET /health` checks the web process only. External service credentials and index
availability are checked when handling requests, so provider outages do not block
web process startup.

## Embedding migration

Uploads and questions use `models/gemini-embedding-001` with 768 dimensions and the
`gemini-embedding-001-768-v1` namespace. **Re-upload your PDFs after upgrading.**
Old vectors are retained but are not searched because embeddings from different
models cannot be mixed. An existing index must have 768 dimensions; otherwise set
`PINECONE_INDEX_NAME` to a new name. The first upload creates a missing index.
Pinecone writes can take a short time to become searchable.

## Local development

Use Python 3.12, create a virtual environment, and install
`server/requirements.txt`. Copy `server/.env.example` to `server/.env` and fill in
your keys. From `server`, run `uvicorn main:app --reload`.

Install `client/requirements.txt`, set `API_URL` to your backend URL (for example
`http://127.0.0.1:8000`), and run `streamlit run client/app.py`.

API routes:
- `POST /upload_pdfs/`: multipart `files` (up to 10 PDFs, 20 MB each).
- `POST /ask/`: form field `question`.
- `GET /health`: process health.

This application currently uses a shared document collection and has no user
authentication or per-user document isolation.

## Regression checks

Install `httpx` alongside the backend dependencies and run:
`python -m unittest discover -s server/tests -v`.
Provider calls are mocked; these checks do not use credentials or mutate Pinecone.
