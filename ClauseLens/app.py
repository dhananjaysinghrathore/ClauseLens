
import os, json, pickle
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from rerank import CrossEncoderReranker

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(APP_DIR, "config.json"), "r", encoding="utf-8"))

INDEX_DIR = os.path.join(APP_DIR, "index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

# --- Load metadata
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

meta = load_jsonl(META_PATH) if os.path.exists(META_PATH) else []
texts = [m["text"] for m in meta]

# --- Dense
emb_model = SentenceTransformer(CFG["EMBED_MODEL"])
index = faiss.read_index(FAISS_PATH) if os.path.exists(FAISS_PATH) else None

# --- BM25
with open(BM25_PATH, "rb") as f:
    bm25_blob = pickle.load(f)
bm25 = bm25_blob["bm25"]

# --- Reranker
reranker = CrossEncoderReranker(CFG["RERANK_MODEL"], device=CFG["DEVICE"])

# --- QA (extractive)
qa = pipeline("question-answering", model=CFG["QA_MODEL"], device=0 if CFG["DEVICE"]=="cuda" else -1)

app = FastAPI(title="ClauseLens RAG API", version="0.1.0")

class AskPayload(BaseModel):
    query: str
    mode: str = "extractive"  # "extractive" | "abstractive"
    top_k: int = 6

def dense_search(query: str, k: int = 12) -> List[int]:
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(q_emb, dtype=np.float32), k)
    return I[0].tolist()

def lexical_search(query: str, k: int = 12) -> List[int]:
    toks = query.lower().split()
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[::-1][:k]
    return idx.tolist()

def hybrid_candidates(query: str, k_dense=12, k_lex=12) -> List[Dict[str, Any]]:
    di = dense_search(query, k_dense)
    li = lexical_search(query, k_lex)
    seen = set()
    out = []
    for i in list(di) + list(li):
        if i in seen: 
            continue
        seen.add(i)
        out.append(meta[i].copy())
    return out

def format_citation(chunk: Dict[str, Any]) -> str:
    return f'{chunk.get("source")} p.{chunk.get("page_start")}-{chunk.get("page_end")} | {chunk.get("section_hint","")[:80]}'

@app.post("/ask")
def ask(p: AskPayload):
    assert index is not None and meta, "Index missing — run ingest.py first."
    # 1) Retrieve hybrid candidates
    cands = hybrid_candidates(p.query, CFG["TOP_K_CANDIDATES"], CFG["TOP_K_CANDIDATES"])
    # 2) Rerank
    top = reranker.rerank(p.query, cands, top_k=CFG["TOP_K_RERANKED"])

    # 3) Extractive QA over top contexts (pool multiple spans)
    answers = []
    for ch in top:
        ans = qa(question=p.query, context=ch["text"], top_k=1, handle_impossible_answer=True)
        if isinstance(ans, list): ans = ans[0]
        score = float(ans.get("score", 0.0))
        if score >= CFG["MIN_ANSWER_SCORE"] and ans.get("answer"):
            answers.append({
                "answer": ans["answer"],
                "score": score,
                "citation": format_citation(ch)
            })
    # If nothing confident, abstain
    if not answers:
        return {
            "query": p.query,
            "mode": p.mode,
            "status": "abstain",
            "message": "I can’t find a clear clause in the indexed documents. Refine the query or ingest more sources.",
            "top_citations": [format_citation(ch) for ch in top]
        }

    # Merge unique spans; keep top 3
    uniq = {}
    for a in answers:
        key = a["answer"].strip()
        if key not in uniq or a["score"] > uniq[key]["score"]:
            uniq[key] = a
    final = sorted(uniq.values(), key=lambda x: x["score"], reverse=True)[:3]

    # 4) Optional: abstractive mode would summarize these, but we keep extractive by default
    if p.mode == "extractive":
        joined = "; ".join([f'"{a["answer"]}" [{a["citation"]}]' for a in final])
        return {
            "query": p.query,
            "answer_mode": "extractive",
            "quotes": final,
            "joined": joined
        }
    else:
        # simple summary without external hallucination: concatenate contexts from top docs
        combined_ctx = "\n\n".join([t["text"] for t in top])
        # For brevity, we just return contexts; hooking a local HF LLM is shown in README.
        return {
            "query": p.query,
            "answer_mode": "abstractive",
            "note": "Attach a local HF LLM to paraphrase 'combined_ctx' cautiously.",
            "top_citations": [format_citation(ch) for ch in top],
            "context_preview": combined_ctx[:1200] + ("..." if len(combined_ctx) > 1200 else "")
        }
