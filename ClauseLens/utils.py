
import re, os, json
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from tqdm import tqdm

def clean_text(txt: str) -> str:
    txt = re.sub(r'\xa0', ' ', txt)
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{2,}', '\n\n', txt)
    return txt.strip()

def extract_pdf_chunks(path: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Parse PDF, keep page numbers, chunk by headings + fixed window fallback."""
    doc = fitz.open(path)
    chunks = []
    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text")
        text = clean_text(text)
        if not text:
            continue
        # simple heading split heuristic
        pieces = re.split(r'\n(?=[A-Z][A-Z0-9 .\-:]{4,}\n)', text)
        if len(pieces) == 1:
            pieces = [text]
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            # sliding window over piece
            start = 0
            while start < len(piece):
                end = min(len(piece), start + chunk_size)
                chunk_txt = piece[start:end]
                chunks.append({
                    "text": chunk_txt,
                    "page_start": pno + 1,
                    "page_end": pno + 1,
                    "source": os.path.basename(path),
                    "section_hint": piece[:120].replace("\n", " ")
                })
                if end == len(piece):
                    break
                start = end - overlap
    doc.close()
    return chunks

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
