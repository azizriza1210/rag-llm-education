import streamlit as st
import requests
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Chatbot PDF",
    page_icon="assets/book.png",
    layout="wide"
)

# --- Inisialisasi Session State ---
# Session state digunakan untuk menyimpan history chat dan status upload
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

# --- Fungsi untuk Menampilkan History Chat ---
def display_chat_history():
    """Menampilkan seluruh pesan dari history chat."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Sidebar untuk Upload PDF ---
with st.sidebar:
    st.header("📁 Upload Modul PDF")
    
    # Widget untuk upload file, hanya menerima .pdf
    uploaded_file = st.file_uploader(
        "Pilih file PDF",
        type='pdf',
        key="pdf_uploader"
    )
    
    # Tombol untuk memicu proses upload
    if st.button("Upload & Proses", key="upload_button"):
        if uploaded_file is not None:
            # Tampilkan spinner saat proses upload berlangsung
            with st.spinner("Mengupload dan memproses PDF..."):
                try:
                    # API endpoint untuk upload
                    upload_url = "https://rag-llm-education.onrender.com/upload-module"
                    
                    # Siapkan file untuk dikirim dalam request
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
                    
                    # Kirim request POST ke API
                    response = requests.post(upload_url, files=files, timeout=60)
                    
                    print(f"Response: {response.text}")
                    # Periksa respons dari server
                    if response.status_code == 200 and response.json().get("message") == "Module uploaded and processed successfully.":
                        st.success("✅ PDF berhasil diupload dan diproses!")
                        st.session_state.pdf_uploaded = True
                        # Kosongkan history chat saat PDF baru diupload
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

# Tampilkan pesan informasi jika PDF belum diupload
if not st.session_state.pdf_uploaded:
    st.info("Silakan upload sebuah modul PDF di sidebar untuk memulai chat.")

# Tampilkan history chat yang sudah ada
display_chat_history()

# Input untuk pertanyaan user di bagian bawah
if prompt := st.chat_input("Tanyakan sesuatu tentang PDF..."):
    # Pastikan PDF sudah diupload sebelum mengajukan pertanyaan
    if not st.session_state.pdf_uploaded:
        st.warning("Anda harus mengupload PDF terlebih dahulu sebelum bertanya.")
        st.stop()

    # Tambahkan pertanyaan user ke history dan tampilkan
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan jawaban dari bot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Bot sedang berpikir..."):
            try:
                # API endpoint untuk chatbot
                chat_url = "https://rag-llm-education.onrender.com/chatbot"
                
                # Payload untuk request
                payload = {"question": prompt}
                
                # Kirim request POST ke API
                response = requests.post(chat_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Maaf, saya tidak bisa menemukan jawaban.")
                    sources = data.get("sources", [])
                    
                    # Format jawaban dan sumber
                    full_response = f"{answer}\n\n"
                    if sources:
                        full_response += "**Sumber:**\n"
                        for source in sources:
                            full_response += f'- <img src="/assets/pdf.png" width="16"> {source}\n'
                    
                    # Tampilkan jawaban akhir
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                else:
                    error_message = f"Maaf, terjadi kesalahan saat menghubungi bot. Status: {response.status_code}"
                    message_placeholder.error(error_message)
                    full_response = error_message

            except requests.exceptions.RequestException as e:
                error_message = f"Maaf, tidak dapat terhubung ke server bot. Error: {e}"
                message_placeholder.error(error_message)
                full_response = error_message
        
        # Tambahkan jawaban bot ke history
        if full_response:
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
