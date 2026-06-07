import re
from typing import Union
from pathlib import Path


def load_document(file_bytes: bytes, file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".docx":
        return _load_docx(file_bytes)
    else:
        raise ValueError(f"Formato não suportado: {ext}. Use PDF, TXT ou DOCX.")


def _load_pdf(file_bytes: bytes) -> str:
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _load_docx(file_bytes: bytes) -> str:
    import io
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping word chunks for better retrieval coverage."""
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks
