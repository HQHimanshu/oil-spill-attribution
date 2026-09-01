"""
Document Ingestion and Semantic Chunking Pipeline for OceanGuard AI RAG System.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .vector_store import DocumentChunk, LocalVectorStore

DOCS_DIR = Path(__file__).resolve().parent / "documents"


def parse_markdown_metadata(content: str) -> dict:
    metadata = {
        "source": "OceanGuard Technical Documentation",
        "title": "Technical Document",
        "date": "2024-01-01",
        "document_type": "Technical Guide",
        "url": "https://oceanguard.ai"
    }
    
    for line in content.splitlines()[:15]:
        line_clean = line.strip()
        if line_clean.startswith("**Source**:") or line_clean.startswith("Source:"):
            metadata["source"] = line_clean.split(":", 1)[1].replace("*", "").strip()
        elif line_clean.startswith("**Title**:") or line_clean.startswith("Title:"):
            metadata["title"] = line_clean.split(":", 1)[1].replace("*", "").strip()
        elif line_clean.startswith("**Date**:") or line_clean.startswith("Date:"):
            metadata["date"] = line_clean.split(":", 1)[1].replace("*", "").strip()
        elif line_clean.startswith("**Document Type**:") or line_clean.startswith("Document Type:"):
            metadata["document_type"] = line_clean.split(":", 1)[1].replace("*", "").strip()
        elif line_clean.startswith("**URL**:") or line_clean.startswith("URL:"):
            metadata["url"] = line_clean.split(":", 1)[1].replace("*", "").strip()

    return metadata


def chunk_markdown_document(doc_path: Path) -> List[DocumentChunk]:
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata = parse_markdown_metadata(text)
    
    # Split by section headers (##)
    sections = re.split(r"\n(?=##\s+)", text)
    chunks: List[DocumentChunk] = []

    for idx, sec in enumerate(sections):
        sec_clean = sec.strip()
        if not sec_clean:
            continue
        
        # Extract section title
        first_line = sec_clean.splitlines()[0]
        sec_title = first_line.replace("#", "").strip() if first_line.startswith("#") else f"Section {idx+1}"
        
        chunk_id = f"{doc_path.stem}_chunk_{idx}"
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=sec_clean,
            source=metadata["source"],
            title=f"{metadata['title']} - {sec_title}",
            document_type=metadata["document_type"],
            date=metadata["date"],
            url=metadata["url"],
            section=sec_title
        )
        chunks.append(chunk)

    return chunks


def build_knowledge_base() -> LocalVectorStore:
    store = LocalVectorStore()
    all_chunks = []

    if DOCS_DIR.exists():
        for md_file in DOCS_DIR.glob("*.md"):
            chunks = chunk_markdown_document(md_file)
            all_chunks.extend(chunks)

    store.add_chunks(all_chunks)
    return store


_GLOBAL_VECTOR_STORE: LocalVectorStore | None = None

def get_vector_store() -> LocalVectorStore:
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        _GLOBAL_VECTOR_STORE = build_knowledge_base()
    return _GLOBAL_VECTOR_STORE
