from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.utils.upload_modules_utils import pdf_chunking_and_store
from src.utils.model_utils import llm, embeddings
import os
import colorlog
from pathlib import Path
from config import CHROMA_PERSIST_DIRECTORY, TEMP_DIR, COLLECTION_NAME, GROQ_API_KEY
import shutil
from src.utils.chatbot_utils import ask_question

# Setup colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(colorlog.INFO)

# Inisialisasi FastAPI
app = FastAPI(title="Simple API", version="1.0.0")

# llm = get_llm(GROQ_API_KEY)
# logger.info("LLM initialized!")

# embeddings = get_embeddings()
# logger.info("Embeddings initialized!")

# Endpoint GET
@app.get("/")
def root():
    return {"message": "API berhasil dijalankan"}

# Endpoint POST
@app.post("/upload-module")
async def upload_module(file: UploadFile = File(...)):
    module_path = None
    
    try:
        # Simpan file ke folder temp
        module_path = TEMP_DIR / file.filename
        logger.debug(f"Menyimpan file: {file.filename}")
        
        with open(module_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File tersimpan di: {module_path}")

        hasil = pdf_chunking_and_store(module_path, embeddings, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME)
        
        if module_path and os.path.exists(module_path):
            os.remove(module_path)
            logger.info(f"File dihapus: {module_path}")

        return hasil
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
class QuestionRequest(BaseModel):
    question: str

@app.post("/chatbot")
async def upload_module(request: QuestionRequest):
    try:
        answer, sources = ask_question(request.question)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))