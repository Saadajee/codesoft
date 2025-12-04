# rag/loader.py
from typing import List
import PyPDF2

def load_txt(path: str) -> str:
    """Load plain text file and return as one string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    import PyPDF2
    text_parts = []

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            text_parts.append(" ".join(t.split()))

    return "\n".join(text_parts)[:1000000]

