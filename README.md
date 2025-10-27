

# 📚 Chatbot Education: RAG-Powered Learning Assistant

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://chatbot-rag-education.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/Architecture-RAG-purple?style=for-the-badge)](https://arxiv.org/abs/2005.11401)

<div align="center">
  <img src="assets/image.png" alt="Chatbot Education Demo" width="800">
</div>

## 🎯 Problem Statement

Many students spend excessive time reading educational modules with numerous pages or avoid studying them altogether due to the overwhelming amount of content.

## 💡 Solution

A Retrieval-Augmented Generation (RAG) chatbot that can answer any questions about educational modules, helping students streamline their learning process.

## ✨ Key Features

- **PDF Processing**: Automatically processes educational modules in PDF format
- **Hierarchical Chunking**: Intelligently splits documents by chapters and sub-chapters
- **Semantic Search**: Finds relevant content based on meaning, not just keywords
- **Context-Aware Responses**: Generates answers based on retrieved document content
- **Hallucination Prevention**: Only answers when relevant information is found
- **Educational Persona**: Responds as a knowledgeable professor

## 🏗️ Architecture

<div align="center">
  <img src="assets/Architecture RAG.jpg" alt="Architecture Diagram" width="600">
</div>

The chatbot is built using a **Retrieval-Augmented Generation (RAG)** architecture with:

- **LangChain**: Framework for LLM applications
- **Groq**: High-performance LLM API
- **ChromaDB**: Open-source vector database
- **HuggingFace**: Local embeddings model (all-MiniLM-L6-v2)

## 🚀 Quick Start

### Option 1: Try the Demo

Visit the live demo: [https://chatbot-rag-education.streamlit.app/](https://chatbot-rag-education.streamlit.app/)

You can ask questions directly about data science topics.

### Option 2: Run Locally

```bash
# Clone the repository
git https://github.com/azizriza1210/rag-llm-education.git
cd rag-llm-education

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
touch .env
```

#### Get API Keys

**Groq API Key (Free)**
1. Visit: [https://console.groq.com](https://console.groq.com)
2. Sign up and log in
3. Navigate to "API Keys"
4. Create a new API key
5. Add to your `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

**HuggingFace API Key (Free)**
1. Visit: [https://huggingface.co/](https://huggingface.co/)
2. Sign up and log in
3. Click your profile, then Settings
4. Create a new API key
5. Add to your `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here
   ```

#### Run the Application

```bash
streamlit run streamlit/app.py
```

## 🔧 How It Works

1. **Document Loading**: PDF documents are loaded into the system
2. **Hierarchical Chunking**: Documents are split by chapters and sub-chapters for better context preservation
3. **Embedding Generation**: Text chunks are converted to embeddings using all-MiniLM-L6-v2 model
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Query Processing**: When a user asks a question:
   - The question is embedded
   - Similar documents are retrieved from the vector database
   - If no relevant documents are found, the chatbot indicates it doesn't have enough information
   - If relevant documents are found, they are used to generate a response

## 📝 Prompt Engineering

The chatbot uses carefully engineered prompts based on the paper ["Exploring Prompt Engineering Practices in the Enterprise"](https://arxiv.org/abs/2403.08950) and ["Does Prompt Formatting Have Any Impact on LLM Performance?"](https://arxiv.org/abs/2411.10541) :

### System Prompt
```
instructions:
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
```

### User Prompt
```
context:
  relevant documents: "{docs}"
  question: "{query}"
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Groq API Key invalid** | Verify API key in `.env` file and check quota in Groq console |
| **ChromaDB errors** | Delete the `chroma_db/` folder and restart the application |
| **Slow embedding model** | The model is downloaded on first run (≈400MB) and then cached locally |
| **No relevant documents found** | Try rephrasing your question or check if the information exists in the uploaded PDF |

## 📖 References

- [LangChain Documentation](https://python.langchain.com/)
- [Groq Console](https://console.groq.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Exploring Prompt Engineering Practices](https://arxiv.org/abs/2403.08950)
- [Does Prompt Formatting Have Any Impact on LLM Performance?](https://arxiv.org/abs/2411.10541)

## 👤 Author

**Your Name**
- GitHub: [@azizriza1210](https://github.com/azizriza1210)
- LinkedIn: [Mohammad Aziz Riza](https://www.linkedin.com/in/mohammad-aziz-riza-3a3862258/)

## 🙏 Acknowledgments

- The open-source community for the amazing tools and libraries
- Groq for providing fast LLM inference
- HuggingFace for the embedding models
- The authors of the referenced papers for their valuable research

---

⭐ If this project helped you, please consider giving it a star!