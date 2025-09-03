
from typing import List, Dict
from FlagEmbedding import FlagReranker

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.reranker = FlagReranker(model_name, use_fp16=(device=="cuda"))

    def rerank(self, query: str, docs: List[Dict], top_k: int = 6) -> List[Dict]:
        pairs = [[query, d["text"]] for d in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        # Attach and sort
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return docs
