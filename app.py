"""
RAG Chatbot dengan Streamlit - Upload PDF
File: app.py

Jalankan dengan: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

# Import LangChain components
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(
    page_title="RAG Chatbot - PDF Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .upload-text {
        font-size: 0.9rem;
        color: #666;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Functions
@st.cache_resource
def get_embeddings():
    """Load embeddings model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_pdf(uploaded_file):
    """Load PDF from uploaded file"""
    try:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return documents
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

def process_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vectorstore(splits, embeddings):
    """Create or update vector store"""
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=None  # In-memory for session
    )
    return vectorstore

def create_rag_chain(vectorstore, groq_api_key, model_name, temperature):
    """Create RAG chain"""
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=1024
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    template = """Kamu adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

Konteks dari dokumen:
{context}

Pertanyaan: {question}

Instruksi:
- Jawab berdasarkan konteks yang diberikan
- Jika informasi tidak ada dalam konteks, katakan "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diupload"
- Berikan jawaban yang jelas, ringkas, dan mudah dipahami
- Gunakan bahasa Indonesia yang baik dan benar

Jawaban:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    
    # API Key
    st.subheader("🔑 Groq API Key")
    groq_api_key = st.text_input(
        "Masukkan Groq API Key",
        type="password",
        help="Dapatkan API key gratis di https://console.groq.com"
    )
    
    if not groq_api_key:
        st.warning("⚠️ Masukkan API key untuk melanjutkan")
        st.markdown("[🔗 Dapatkan API Key Gratis](https://console.groq.com/keys)")
    
    st.divider()
    
    # Model selection
    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "Pilih Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        help="llama-3.1-8b-instant: Paling cepat\nllama-3.1-70b-versatile: Lebih pintar"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Semakin rendah = lebih fokus dan konsisten"
    )
    
    st.divider()
    
    # Chunking settings
    st.subheader("📄 Document Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=2000,
        value=1000,
        step=100,
        help="Ukuran potongan teks (karakter)"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap antar chunk untuk menjaga konteks"
    )
    
    st.divider()
    
    # Processed files
    if st.session_state.processed_files:
        st.subheader("📚 File yang Diproses")
        for file in st.session_state.processed_files:
            st.text(f"✅ {file}")
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear All Data"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.processed_files = []
        st.rerun()

# ========== MAIN CONTENT ==========
st.title("🤖 RAG Chatbot - PDF Q&A")
st.markdown("Upload dokumen PDF dan tanyakan apapun tentang isinya!")

# File uploader
st.subheader("📤 Upload Dokumen PDF")
uploaded_files = st.file_uploader(
    "Pilih file PDF (bisa multiple files)",
    type=['pdf'],
    accept_multiple_files=True,
    help="Upload satu atau lebih file PDF untuk diproses"
)

# Process uploaded files
if uploaded_files and groq_api_key:
    if st.button("🚀 Process Documents"):
        with st.spinner("⏳ Memproses dokumen..."):
            try:
                all_documents = []
                
                # Load all PDFs
                progress_bar = st.progress(0)
                for idx, uploaded_file in enumerate(uploaded_files):
                    st.info(f"📖 Loading: {uploaded_file.name}")
                    documents = load_pdf(uploaded_file)
                    if documents:
                        all_documents.extend(documents)
                        if uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if not all_documents:
                    st.error("❌ Tidak ada dokumen yang berhasil diload")
                else:
                    # Split documents
                    st.info(f"✂️ Splitting {len(all_documents)} pages into chunks...")
                    splits = process_documents(all_documents, chunk_size, chunk_overlap)
                    st.success(f"✅ Created {len(splits)} chunks")
                    
                    # Create embeddings
                    st.info("🔢 Creating embeddings...")
                    embeddings = get_embeddings()
                    
                    # Create vector store
                    st.info("💾 Building vector store...")
                    st.session_state.vectorstore = create_vectorstore(splits, embeddings)
                    
                    # Create RAG chain
                    st.info("🔗 Creating RAG chain...")
                    st.session_state.rag_chain = create_rag_chain(
                        st.session_state.vectorstore,
                        groq_api_key,
                        model_name,
                        temperature
                    )
                    
                    st.success("✅ Dokumen berhasil diproses! Silakan ajukan pertanyaan.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display chat messages
st.divider()
st.subheader("💬 Chat")

# Chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if st.session_state.rag_chain and groq_api_key:
    if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    if not groq_api_key:
        st.info("👈 Masukkan Groq API Key di sidebar untuk memulai")
    elif not st.session_state.rag_chain:
        st.info("👆 Upload dan process dokumen PDF terlebih dahulu")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 Powered by LangChain + Groq + ChromaDB | 
        <a href='https://console.groq.com' target='_blank'>Get Free API Key</a></p>
    </div>
""", unsafe_allow_html=True)