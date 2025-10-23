import pymupdf4llm
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

def pdf_chunking_hierarchical(pdf_path: str):
    # Ekstrak nama file dari path
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
            metadata = chunk.metadata.copy()
            metadata["source"] = pdf_path
            metadata["file_name"] = file_name
            
            final_chunks.append(
                Document(
                    page_content=sub,
                    metadata=metadata
                )
            )

    return final_chunks