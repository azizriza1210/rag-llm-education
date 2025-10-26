import pymupdf4llm
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME

def pdf_chunking_hierarchical(pdf_path: str):
    file_name = os.path.basename(pdf_path)
    
    md_text = pymupdf4llm.to_markdown(pdf_path)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    hierarchical_chunks = markdown_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    final_chunks = []
    for chunk in hierarchical_chunks:
        content = chunk.page_content
        sub_chunks = text_splitter.split_text(content)

        for sub in sub_chunks:
            # Gabungkan metadata existing dengan nama file
            metadata = chunk.metadata.copy()
            metadata["source"] = pdf_path  # path lengkap
            metadata["file_name"] = file_name  # nama file saja
            
            final_chunks.append(
                Document(
                    page_content=sub,
                    metadata=metadata
                )
            )

    return final_chunks

def store_to_chromadb(hierarchical_chunks, embeddings, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME):
    try:
        # Load existing vector store atau buat baru
        try:
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            existing_collection = True
        except:
            vector_store = None
            existing_collection = False
        
        # Cek file_name yang akan ditambahkan
        new_file_name = hierarchical_chunks[0].metadata.get('file_name') if hierarchical_chunks else None
        
        if existing_collection and new_file_name:
            # Cek apakah file sudah ada
            existing_docs = vector_store.get()
            existing_file_names = set()
            
            if existing_docs and existing_docs.get('metadatas'):
                for meta in existing_docs['metadatas']:
                    if meta.get('file_name'):
                        existing_file_names.add(meta['file_name'])
            
            # Jika file sudah ada, hapus data lama
            if new_file_name in existing_file_names:
                print(f"⚠️ File '{new_file_name}' sudah ada. Menghapus data lama...")
                
                # Hapus semua dokumen dengan file_name yang sama
                ids_to_delete = []
                for i, meta in enumerate(existing_docs['metadatas']):
                    if meta.get('file_name') == new_file_name:
                        ids_to_delete.append(existing_docs['ids'][i])
                
                if ids_to_delete:
                    vector_store.delete(ids=ids_to_delete)
                    print(f"✅ Berhasil menghapus {len(ids_to_delete)} dokumen lama dari '{new_file_name}'")
        
        # Tambahkan dokumen baru
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=hierarchical_chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                collection_name=COLLECTION_NAME
            )
        else:
            vector_store.add_documents(hierarchical_chunks)
        
        # Hapus duplikat berdasarkan content dan metadata
        print("🔍 Memeriksa duplikat...")
        all_docs = vector_store.get()
        
        if all_docs and all_docs.get('ids'):
            seen = set()
            ids_to_delete = []
            
            for i, (doc_id, content, metadata) in enumerate(zip(
                all_docs['ids'],
                all_docs['documents'],
                all_docs['metadatas']
            )):
                # Buat signature unik dari content + metadata penting
                signature = (
                    content,
                    metadata.get('file_name'),
                    metadata.get('Header 1'),
                    metadata.get('Header 2'),
                    metadata.get('Header 3')
                )
                
                if signature in seen:
                    ids_to_delete.append(doc_id)
                else:
                    seen.add(signature)
            
            # Hapus duplikat
            if ids_to_delete:
                vector_store.delete(ids=ids_to_delete)
                print(f"🗑️ Berhasil menghapus {len(ids_to_delete)} dokumen duplikat")
        
        total_docs = len(vector_store.get()['ids']) if vector_store.get() else len(hierarchical_chunks)
        
        return f"✅ Berhasil menyimpan {len(hierarchical_chunks)} dokumen baru ke ChromaDB.\n📊 Total dokumen di database: {total_docs}"

    except Exception as e:
        return f"❌ Gagal menyimpan ke ChromaDB: {e}"
    
def pdf_chunking_and_store(module_path: str, embeddings, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME):
    try: 
        hierarchical_chunks = pdf_chunking_hierarchical(module_path)
        hasil = store_to_chromadb(hierarchical_chunks, embeddings, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME)
        print(hasil)
        return "Module uploaded and processed successfully."
    except Exception as e:
        return f"Error during chunking and storing: {e}"