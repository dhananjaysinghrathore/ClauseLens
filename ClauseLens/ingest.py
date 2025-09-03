
import os, json, argparse, pickle
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

from utils import extract_pdf_chunks, save_jsonl

def build_dense_index(texts: List[str], model_name: str, out_path: str):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, out_path)
    np.save(out_path + ".npy", emb.astype(np.float32))
    return out_path

def build_bm25(corpus: List[str], out_path: str):
    tokenized = [t.lower().split() for t in corpus]
    bm25 = BM25Okapi(tokenized)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--embed_model", default="BAAI/bge-m3")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_chunks: List[Dict[str, Any]] = []

    for fname in tqdm(os.listdir(args.data_dir), desc="Scanning data"):
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in [".pdf"]:
            chunks = extract_pdf_chunks(fpath, args.chunk_size, args.overlap)
            all_chunks.extend(chunks)
        # TODO: add HTML loader if needed

    assert all_chunks, "No chunks created — add PDFs to data_dir."
    save_jsonl(os.path.join(args.out_dir, "meta.jsonl"), all_chunks)

    texts = [c["text"] for c in all_chunks]
    print(f"Building dense FAISS on {len(texts)} chunks")
    build_dense_index(texts, args.embed_model, os.path.join(args.out_dir, "faiss.index"))

    print("Building BM25")
    build_bm25(texts, os.path.join(args.out_dir, "bm25.pkl"))

    print("Done. Files written to", args.out_dir)

if __name__ == "__main__":
    main()
