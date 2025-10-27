from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import GROQ_API_KEY, HUGGINGFACE_API_KEY
import os

# def get_llm(GROQ_API_KEY):
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)
    # return llm

# def get_embeddings():
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# embeddings = HuggingFaceEmbeddings(
#     model_name="models/all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
# )
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Model kecil (~25MB)
# embeddings = HuggingFaceEmbeddings(
#     model_name="models/all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
from langchain_huggingface import HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)
    # return embeddings
