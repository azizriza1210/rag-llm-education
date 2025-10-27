import streamlit as st
import requests
import os

# --- Custom CSS untuk Dark Mode ---
st.markdown("""
<style>
    /* Tema Warna Umum */
    :root {
        --bg-color: #0E1117;
        --sidebar-bg-color: #262730;
        --primary-color: #4A90E2; /* Biru yang menenangkan */
        --secondary-color: #F0F2F6; /* Warna teks sekunder */
        --user-msg-bg: #1E3A5F; /* Biru tua untuk pesan user */
        --bot-msg-bg: #2B2D42; /* Abu-abu tua untuk pesan bot */
        --source-bg: #1F2937; /* Latar belakang untuk sumber */
    }

    /* Gaya Latar Belakang Utama */
    .stApp {
        background-color: var(--bg-color);
    }

    /* Gaya Sidebar */
    .css-1d391kg { /* Perhatikan selector ini bisa berubah, gunakan inspect element jika tidak bekerja */
        background-color: var(--sidebar-bg-color);
    }
    .css-1d391kg h1 {
        color: var(--primary-color);
        font-family: 'Monospace', sans-serif;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }

    /* Gaya Tombol Upload */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Gaya Judul Utama */
    .stTitle {
        color: var(--secondary-color);
        font-family: 'Monospace', sans-serif;
        text-align: center;
    }

    /* Gaya Pesan Chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage[data-testid="chat-message-container-user"] {
        background-color: var(--user-msg-bg);
    }
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background-color: var(--bot-msg-bg);
    }

    /* Gaya Container untuk Sumber */
    .sources-container {
        background-color: var(--source-bg);
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 5px;
    }
    .sources-container p {
        margin: 0;
    }

    /* Gaya untuk Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: var(--secondary-color);
    }
    .empty-state h2 {
        color: var(--primary-color);
    }

    /* Gaya Input Chat */
    .stChatInput > div > div > input {
        background-color: #2B2D42;
        color: var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Chatbot PDF",
    page_icon="assets/book.png",
    layout="wide"
)

# --- Inisialisasi Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

# --- Fungsi untuk Menampilkan History Chat ---
def display_chat_history():
    """Menampilkan seluruh pesan dari history chat."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Gunakan unsafe_allow_html untuk merender div kustom
            st.markdown(message["content"], unsafe_allow_html=True)

# --- Sidebar untuk Upload PDF ---
with st.sidebar:
    st.header("📁 Upload Modul PDF")
    
    uploaded_file = st.file_uploader(
        "Pilih file PDF",
        type='pdf',
        key="pdf_uploader"
    )
    
    # Tambahkan emoji pada tombol untuk visual yang lebih baik
    if st.button("📤 Upload & Proses", key="upload_button"):
        if uploaded_file is not None:
            with st.spinner("Mengupload dan memproses PDF..."):
                try:
                    upload_url = "https://rag-llm-education.onrender.com/upload-module"
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                    response = requests.post(upload_url, files=files, timeout=60)
                    
                    if response.status_code == 200 and response.json().get("message") == "Module uploaded and processed successfully.":
                        st.success("✅ PDF berhasil diupload dan diproses!")
                        st.session_state.pdf_uploaded = True
                        st.session_state.chat_history = []
                    else:
                        st.error(f"❌ Gagal mengupload PDF. Status: {response.status_code}, Pesan: {response.text}")
                        st.session_state.pdf_uploaded = False

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Terjadi kesalahan saat menghubungi server: {e}")
                    st.session_state.pdf_uploaded = False
        else:
            st.warning("Silakan pilih file PDF terlebih dahulu.")

# --- Area Chat Utama ---
st.title("💬 Tanyakan Apa Saja Tentang PDF Anda")

# Tampilan awal yang lebih menarik jika PDF belum diupload
if not st.session_state.pdf_uploaded:
    st.markdown("""
    <div class="empty-state">
        <h2>Selamat Datang! 👋</h2>
        <p>Anda dapat langsung bertanya atau upload modul PDF anda sendiri di sidebar sebelah kiri.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Tampilkan history chat yang sudah ada
    display_chat_history()

# Input untuk pertanyaan user di bagian bawah
if prompt := st.chat_input("Tanyakan sesuatu tentang PDF..."):
    # Anda bisa membatalkan komentar di bawah jika ingin memaksa upload sebelum chat
    # if not st.session_state.pdf_uploaded:
    #     st.warning("Anda harus mengupload PDF terlebih dahulu sebelum bertanya.")
    #     st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Bot sedang berpikir..."):
            try:
                chat_url = "https://rag-llm-education.onrender.com/chatbot"
                payload = {"question": prompt}
                response = requests.post(chat_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Maaf, saya tidak bisa menemukan jawaban.")
                    sources = data.get("sources", [])
                    
                    # Bungkus jawaban dalam div
                    response_html = f"<p>{answer}</p>"
                    
                    # Bungkus sumber dalam div dengan khusus
                    if sources:
                        sources_html = "<div class='sources-container'><p><strong>Sumber:</strong></p><ul>"
                        for source in sources:
                            sources_html += f'<li>📄 {source}</li>'
                        sources_html += "</ul></div>"
                        response_html += sources_html
                    
                    # Tampilkan jawaban akhir
                    message_placeholder.markdown(response_html, unsafe_allow_html=True)
                    full_response = response_html
                else:
                    error_message = f"Maaf, terjadi kesalahan saat menghubungi bot. Status: {response.status_code}"
                    message_placeholder.error(error_message)
                    full_response = f"<p style='color: red;'>{error_message}</p>"

            except requests.exceptions.RequestException as e:
                error_message = f"Maaf, tidak dapat terhubung ke server bot. Error: {e}"
                message_placeholder.error(error_message)
                full_response = f"<p style='color: red;'>{error_message}</p>"
        
        if full_response:
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
