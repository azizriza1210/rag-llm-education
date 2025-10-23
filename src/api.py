from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from src.store_data_to_chroma import pdf_chunking_hierarchical
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import colorlog
from pathlib import Path

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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PERSIST_DIRECTORY = "../chroma_db"
COLLECTION_NAME = "machine_learning_modules"
TEMP_DIR = Path("data/documents")
TEMP_DIR.mkdir(exist_ok=True)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)
logger.info("LLM initialized!")

# Model untuk request POST
# class NameRequest(BaseModel):
#     nama: str

# Endpoint GET
@app.get("/")
def root():
    return {"message": "API berhasil dijalankan"}

# Endpoint POST
@app.post("/upload-module")
async def upload_module(file: UploadFile = File(...)):
    file_path = None
    
    try:
        # Simpan file ke folder temp
        file_path = TEMP_DIR / file.filename
        logger.debug(f"Menyimpan file: {file.filename}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.debug(f"File tersimpan: {file_path}")
        
        # PROSES FILE DI SINI
        file_size = os.path.getsize(file_path)
        logger.debug(f"Memproses file: {file_size} bytes")
        
        # Your processing logic here...
        result = {
            "filename": file.filename,
            "size": file_size,
            "message": "File berhasil diproses"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # finally:
    #     # Hapus file setelah selesai
    #     if file_path and os.path.exists(file_path):
    #         os.remove(file_path)
    #         logger.info(f"File dihapus: {file_path}")

# Jalankan dengan: uvicorn main:app --reload