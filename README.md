# 📘 ClauseLens — RAG-powered Compliance Chatbot

**ClauseLens** is an open-source **Regulatory & Compliance RAG (Retrieval-Augmented Generation) assistant**.  
It is designed for industries like **energy, legal, and government procurement**, where teams must navigate **hundreds of pages of tariff orders, RFQs, and policies** with absolute accuracy and zero hallucination.  

---

## 🚀 Key Features
- **Document-grounded QA** — Answers are extracted *verbatim* from ingested PDFs.  
- **Hybrid Retrieval** — Combines FAISS (dense embeddings) with BM25 (lexical search).  
- **Cross-Encoder Reranker** — Improves accuracy of retrieved clauses.  
- **Extractive QA Model** — RoBERTa-based QA for word-for-word clause retrieval.  
- **Abstain Mode** — If context is weak, the bot refuses to guess, ensuring compliance.  
- **Local & Secure** — All models run on-premises; no cloud API required (sensitive for government tenders).  
- **FastAPI Backend + Streamlit UI** — Easy deployment and interactive use.  

---

## 🧩 System Prompt (Core Principle)
> *“Always answer using the ingested documents. Provide exact quotes with citations (document name, page, section).  
> If the context is insufficient, abstain instead of guessing. Do not hallucinate.”*

---

## 📂 Architecture Overview
- **Data ingestion** → PDF parsing & chunking (PyMuPDF).  
- **Indexing** → FAISS embeddings + BM25 keyword index.  
- **Retrieval** → Hybrid search + reranking.  
- **QA Layer** → Extractive answer with citations.  
- **Interface** → FastAPI backend + Streamlit chatbot UI.  

---

## 💡 Example Use Cases
- **Energy Sector** — Query CERC/SERC tariff orders, wheeling charges, open access surcharges.  
- **Legal Teams** — Search contracts, compliance manuals clause-by-clause.  
- **Procurement** — Navigate RFQs/tender conditions (bid security, net worth, timelines).  

---

## 📈 Benefits
- Saves **hours of manual reading** of 500+ page regulatory docs.  
- Reduces **compliance risk** by providing verifiable answers.  
- Ensures **data confidentiality** by running locally (crucial for govt tenders).  
- Improves **decision-making speed** for legal & regulatory departments.  

---

## ⚠️ Limitations
- Ingest can be slow on CPU for very large corpora.  
- Limited to ingested documents (no internet knowledge).  
- GPU recommended for faster embedding & reranking.  

---

## 🔮 Future Scope
- Table extraction (tariff schedules, penalty rates).  
- Multilingual support (Hindi/regional regulations).  
- ERP/Dashboard integration for enterprises.  
- Auto-ingestion of new tariff orders & notifications.  

---

## 🛠️ Tech Stack
- **Python** (FastAPI, Streamlit, PyMuPDF, FAISS, BM25, Transformers)  
- **Models**:  
  - Embeddings → `BAAI/bge-m3`  
  - Reranker → `BAAI/bge-reranker-v2-m3`  
  - QA → `deepset/roberta-base-squad2`  
  - (Optional summarizer: Llama 3 / Qwen, local only)  

---
