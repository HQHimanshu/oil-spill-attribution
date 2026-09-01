"""
Local Vector Store and Semantic Retrieval Engine for OceanGuard AI RAG System.
Provides text chunk indexing, TF-IDF / dense semantic vector encoding, cosine similarity ranking, and metadata filtering.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        title: str,
        document_type: str,
        date: str,
        url: str,
        section: Optional[str] = None
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.title = title
        self.document_type = document_type
        self.date = date
        self.url = url
        self.section = section or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "document_type": self.document_type,
            "date": self.date,
            "url": self.url,
            "section": self.section
        }


class LocalVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            return
        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.05:  # Relevance threshold
                results.append((self.chunks[idx], round(score, 4)))

        return results
