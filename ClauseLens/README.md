# ClauseLens — RAG for Policy & Tariff Orders (Local, No-Internet Mode)

**Goal:** An on‑prem, citation‑first chatbot for regulatory/compliance teams in energy & legal sectors.  
**Design:** Extractive‑by‑default (verbatim quotes + page/section refs) with optional summarization. No web calls during QA.

## What you get
- **Ingestion**: PDF/HTML loaders, OCR flag, heading‑aware chunking, metadata (page/section/source).
- **Hybrid Retrieval**: FAISS (dense, `bge-m3`) + BM25 (lexical) → **Cross‑encoder rerank** (`bge-reranker-v2-m3`).
- **Answering**: 
  - **Extractive**: QA model finds word‑for‑word spans.
  - **Abstractive (optional)**: local HF LLM to paraphrase *only using retrieved text*.
- **Guardrails**: If confidence/citations are weak → **abstain** with “cannot find in corpus”.

## Quick start
### 0) Python & GPU
- Python 3.10–3.11 recommended.
- GPU optional. CPU will work for PoC (slower). For local LLM generation, a CUDA GPU helps.

### 1) Install deps
```bash
pip install -r requirements.txt
```

> If Windows + no CUDA: it still works for extractive QA + retrieval. Abstractive generation is optional.

### 2) Put documents
Drop PDFs (tariff orders, policies) into:  
```
/mnt/data/clauselens_rag/data
```

### 3) Build indexes
```bash
python ingest.py --data_dir ./data --out_dir ./index
```
This will:
- Parse PDFs (PyMuPDF), keep page numbers & headings
- Create FAISS dense index with `BAAI/bge-m3`
- Create BM25 index
- Save `meta.json` with chunk metadata

### 4) Run API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Test:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query":"What is the wheeling charge for 11 kV per unit in Order XYZ?", "mode":"extractive"}'
```

### 5) (Optional) Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Model choices (Hugging Face)
- **Embeddings**: `BAAI/bge-m3` (multilingual, great for laws/policies)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (cross‑encoder)
- **Extractive QA**: `deepset/roberta-base-squad2` (fast, robust); swap to `deepset/bert-large-uncased-whole-word-masking-squad2` if GPU memory is ample
- **Abstractive LLM (optional)**: e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` via Transformers (can run 4‑bit with bitsandbytes on Linux; otherwise CPU fallback).

> For strict “no hallucination,” keep **extractive mode** as default and only show verbatim quotes + citations.

## Architecture
1. **Ingest**: parse → clean → chunk (heading‑aware + overlap) → store chunks + metadata
2. **Retrieve**: FAISS (top_k) ∪ BM25 (top_k) → **cross‑encoder rerank**
3. **Answer**:
   - **Extractive**: QA model extracts spans from top reranked chunks. We stitch them + show citations (doc, page, section).
   - **Abstractive (optional)**: LLM summarizes **with strict instruction to only use provided chunks**; if insufficient → abstain.
4. **Guardrails**: score thresholds (similarity + reranker + QA score).

## Tuning knobs
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: start at 1200/200 chars for policies; increase if many long clauses.
- `HYBRID_ALPHA`: weight between FAISS cosine and BM25 score (default 0.6 dense / 0.4 lexical).
- `TOP_K_CANDIDATES`: 8–16, `TOP_K_RE_RANKED`: 4–8.
- QA `MIN_ANSWER_SCORE` = 0.25–0.35 to avoid over‑confident wrong spans.

## Security & Compliance
- Local only. No outside calls during `/ask`.
- Citations include **source filename + page range + heading**.
- “Cannot answer with confidence” fallback when retrieved context is weak.

## Roadmap
- Table‑aware extraction (camelot/tabula) with structured retrieval
- PDF coordinate → clickable preview (highlight clause)
- Scheduler to re‑ingest new orders (cron)
