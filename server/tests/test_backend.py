import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import UploadFile
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from main import app
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from routes.ask_question import SimpleRetriever


class BackendTests(unittest.TestCase):
    def test_startup_without_credentials_or_network(self):
        env = {k: v for k, v in os.environ.items() if not k.endswith('API_KEY')}
        env['PYTHON_DOTENV_DISABLED'] = '1'
        code = '''
from unittest.mock import patch
with patch("socket.socket.connect", side_effect=AssertionError("Network during startup")):
    from main import app
    from fastapi.testclient import TestClient
    assert TestClient(app).get("/health").json() == {"status": "ok"}
'''
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                       env=env, check=True, capture_output=True, text=True)

    def test_upload_stores_text_source_and_cleans_temp_file(self):
        doc = Document(page_content='Document text', metadata={'page': 0})
        with patch('modules.load_vectorstore.PyPDFLoader') as loader, \
             patch('modules.load_vectorstore.get_embeddings') as embeddings, \
             patch('modules.load_vectorstore.get_index') as index:
            loader.return_value.load.return_value = [doc]
            embeddings.return_value.embed_documents.return_value = [[0.1] * 768]
            load_vectorstore([UploadFile(filename='../../report.pdf', file=io.BytesIO(b'%PDF-1.4 fake'))])
            args = index.return_value.upsert.call_args.kwargs
            self.assertEqual(args['vectors'][0]['metadata']['text'], 'Document text')
            self.assertEqual(args['vectors'][0]['metadata']['source'], 'report.pdf')
            self.assertEqual(args['namespace'], 'gemini-embedding-001-768-v1')
            self.assertFalse(Path(loader.call_args.args[0]).exists())

    def test_invalid_upload_and_blank_question(self):
        client = TestClient(app)
        self.assertEqual(client.post('/upload_pdfs/', files={'files': ('bad.pdf', b'bad')}).status_code, 400)
        self.assertEqual(client.post('/ask/', data={'question': '   '}).status_code, 400)

    def test_real_chain_invocation_and_sources(self):
        docs = [Document(page_content='Document context', metadata={'source': 'report.pdf'})]
        with patch('modules.llm.require_env', return_value='fake'), \
             patch('modules.llm.ChatGroq', return_value=FakeListLLM(responses=['Grounded answer'])):
            result = query_chain(get_llm_chain(SimpleRetriever(documents=docs)), 'Question')
        self.assertEqual(result, {'response': 'Grounded answer', 'sources': ['report.pdf']})

    def test_question_retrieves_uploaded_text(self):
        with patch('routes.ask_question.get_index') as index, \
             patch('routes.ask_question.get_embeddings') as embeddings, \
             patch('modules.llm.require_env', return_value='fake'), \
             patch('modules.llm.ChatGroq', return_value=FakeListLLM(responses=['Answer'])):
            index.return_value.query.return_value = {'matches': [
                {'metadata': {'text': 'Context', 'source': 'report.pdf'}}, {'metadata': None}]}
            embeddings.return_value.embed_query.return_value = [0.1] * 768
            response = TestClient(app).post('/ask/', data={'question': 'Question'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'response': 'Answer', 'sources': ['report.pdf']})
            self.assertEqual(index.return_value.query.call_args.kwargs['namespace'],
                             'gemini-embedding-001-768-v1')

    def test_empty_retrieval_does_not_call_llm(self):
        with patch('routes.ask_question.get_index') as index, \
             patch('routes.ask_question.get_embeddings'), \
             patch('routes.ask_question.get_llm_chain') as chain:
            index.return_value.query.return_value = {'matches': []}
            response = TestClient(app).post('/ask/', data={'question': 'Question'})
            self.assertEqual(response.status_code, 200)
            chain.assert_not_called()


if __name__ == '__main__':
    unittest.main()
