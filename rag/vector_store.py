from dataclasses import dataclass
import numpy as np
import faiss


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


class VectorStore:
    """FAISS-based vector store with cosine similarity search."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, texts: list[str], source: str) -> int:
        if not texts:
            return 0

        from rag.embeddings import embed_texts

        embeddings = embed_texts(texts)
        faiss.normalize_L2(embeddings)

        start = len(self.chunks)
        for i, text in enumerate(texts):
            self.chunks.append(Chunk(text=text, source=source, chunk_index=start + i))

        self.index.add(embeddings)
        return len(texts)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self.index.ntotal == 0:
            return []

        from rag.embeddings import embed_query

        q_emb = embed_query(query).reshape(1, -1)
        faiss.normalize_L2(q_emb)

        k = min(k, self.index.ntotal)
        _, indices = self.index.search(q_emb, k)

        return [self.chunks[idx] for idx in indices[0] if idx >= 0]

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    @property
    def total(self) -> int:
        return self.index.ntotal
