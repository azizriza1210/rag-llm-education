from langchain_core.prompts import ChatPromptTemplate
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.utils.model_utils import llm, embeddings

vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
print("Database berhasil dimuat.")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

system_prompt = """instructions:
task: Tugasmu adalah menjawab pertanyaan dari mahasiswa berdasarkan dokumen modul ajar yang diberikan. Gunakan informasi dari dokumen untuk memberikan jawaban yang akurat dan relevan.
persona: Kamu adalah seorang dosen yang menjawab pertanyaan mahasiswa dengan detail dan jelas.
method: Untuk menjawab pertanyaan, ikuti langkah-langkah berikut:
1. Baca pertanyaan mahasiswa dengan seksama.
2. Cari informasi yang relevan dari dokumen modul ajar yang diberikan.
3. Susun jawaban yang komprehensif dan mudah dipahami berdasarkan informasi tersebut.
4. Jika informasi tidak cukup, katakan bahwa kamu tidak memiliki cukup data untuk menjawab pertanyaan tersebut.
output-length: Jawaban harus padat sesuai dengan yang ada di dokumen.
output-format: sebuah paragraf.
inclusion: Penjelasan dari dokumen modul ajar yang relevan dengan pertanyaan.
handle-unknown: Jika informasi yang diberikan tidak cukup untuk menjawab pertanyaan, katakan 'Maaf, saya tidak memiliki cukup informasi untuk menjawab pertanyaan ini.'
"""

user_prompt = """context:
  relevant documents: "{docs}"
  question: "{query}"
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])
print("✅ Prompt template loaded from JSON successfully")


def format_docs(docs):
    """Format dokumen untuk context"""
    return "\n\n".join(doc.page_content for doc in docs)

# Chain menggunakan LCEL (LangChain Expression Language)
rag_chain = (
    {"docs": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG Chain created!")

def ask_question(question):
    """Fungsi untuk bertanya ke chatbot"""
    print(f"Pertanyaan: {question}")
    
    # Dapatkan jawaban dari RAG chain
    answer = rag_chain.invoke(question)
    
    # Dapatkan dokumen sumber dari retriever
    docs = retriever.invoke(question)

    print("📚 Sumber Dokumen:")
    file_names = set()  # untuk menyimpan nama file unik
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        file_name = metadata.get("file_name")
        if file_name:
            file_names.add(file_name)

        print(f"Metadata: {metadata}")
        print(f"Data:\n{i}. {doc.page_content[:200]}...")  # print sebagian aja biar gak panjang

    return answer, list(file_names)


def get_response(question: str) -> dict:
    """Fungsi untuk mendapatkan response dari chatbot"""
    try:
        answer, file_names = ask_question(question)
        return {
            "answer": answer,
            "sources": file_names
        }
    except Exception as e:
        return {
            "error": str(e)
        }
