import requests
from config import API_URL


def upload_pdfs_api(files):
    files_payload=[ ("files",(f.name,f.getvalue(),"application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/",files=files_payload,timeout=(10, 300))

def ask_question(question):
    return requests.post(f"{API_URL}/ask/",data={"question":question},timeout=(10, 120))