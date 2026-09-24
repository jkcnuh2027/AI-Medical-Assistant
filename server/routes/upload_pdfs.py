from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from modules.load_vectorstore import load_vectorstore
from logger import logger

router = APIRouter()


@router.post("/upload_pdfs/")
def upload_pdfs(files: list[UploadFile] = File(...)):
    try:
        if len(files) > 10:
            raise HTTPException(400, "Upload at most 10 PDFs at a time")
        load_vectorstore(files)
        return {"messages": "Files processed and vectorstore updated"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": "Unable to process PDFs"})
    finally:
        for file in files:
            file.file.close()
