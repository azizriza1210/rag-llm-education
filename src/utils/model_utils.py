from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import GROQ_API_KEY

# def get_llm(GROQ_API_KEY):
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)
    # return llm

# def get_embeddings():
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    # return embeddings
