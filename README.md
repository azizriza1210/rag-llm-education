# Chatbot Education

**Problem**: Saat ini banyak mahasiswa yang memakan banyak waktu bahkan malas membaca dan mempelajari modul ajar dengan halaman yang banyak.

**Solusi**: Oleh karena itu, solusi yang dikembangkan berupa chatbot yang mampu menjwab pertanyaan apapun seputar modul ajar yang dapat membantu mahasiswa dalam proses belajar.

Chatbot dikembangkan berbasis **Retrieval-Augmented Generation (RAG)** yang menggunakan:
- **LangChain**: Framework untuk aplikasi LLM
- **Groq**: API LLM
- **ChromaDB**: Vector database open-source
- **HuggingFace**: Embeddings model (Free & local)

## Cara Memulai

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Dapatkan API Key Groq (Free)

1. Kunjungi: https://console.groq.com
2. Daftar & login akun
3. Buka menu "API Keys"
4. Buat API key baru
5. Copy API key ke file `.env` dan simpan pada variabel berikut
```bash
GROQ_API_KEY=API_KEY
```

### 3. Jalankan Notebook
```bash
# Install Jupyter jika belum
pip install jupyter

# Jalankan Jupyter
jupyter notebook notebooks/rag_chatbot.ipynb
```

### 4. Tambahkan Dokumen

- Letakkan file PDF atau TXT di folder `data/documents/`
- Update kode di notebook untuk load dokumen Anda

## 📚 Fitur

- ✅ Load dokumen PDF dan TXT
- ✅ Chunking dokumen otomatis
- ✅ Embeddings menggunakan model lokal (gratis)
- ✅ Vector store persistent dengan ChromaDB
- ✅ Query dengan context dari dokumen
- ✅ Source tracking (tahu jawaban dari dokumen mana)
- ✅ Interactive chat mode

## 💡 Tips Penggunaan

1. **Model Groq yang tersedia (gratis)**:
   - `llama-3.1-8b-instant` (paling cepat)
   - `llama-3.1-70b-versatile` (lebih pintar)
   - `mixtral-8x7b-32768` (context window besar)

2. **Optimize chunking**:
   - Sesuaikan `chunk_size` dan `chunk_overlap`
   - Dokumen teknis: chunk lebih kecil (300-500)
   - Dokumen naratif: chunk lebih besar (1000-1500)

3. **Improve retrieval**:
   - Ubah `k` di `search_kwargs` (jumlah dokumen yang diambil)
   - Coba similarity score threshold

## 🔧 Troubleshooting

**Error: Groq API Key invalid**
- Pastikan API key benar di file `.env`
- Cek quota di console Groq

**Error: ChromaDB**
- Hapus folder `chroma_db/` dan run ulang

**Model embedding slow**
- Model di-download pertama kali (±400MB)
- Setelah itu akan cached secara lokal

## 📖 Referensi

- [LangChain Docs](https://python.langchain.com/)
- [Groq Console](https://console.groq.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)

## 📝 Lisensi

MIT License - Bebas digunakan untuk project pribadi maupun komersial