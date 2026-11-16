"""Retrieval module for Day 7.
Implements:
1. Dense Retrieval (Embedding + Cosine similarity, with optional metadata filter)
2. Lexical Retrieval (BM25 / TF-IDF)
3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
"""
import argparse
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Common helper & text preprocessing ---

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def hit(chunk: dict, gold: list[dict]) -> bool:
    for g in gold:
        if (chunk.get("source_id") == g.get("source_id") and
            chunk.get("page") == g.get("page") and
            g.get("evidence", "") in chunk.get("text", "")):
            return True
    return False


def build_context(ranked_chunks: list[dict], budget: int = 800) -> tuple[str, list[dict]]:
    """Accumulate chunks into context string while respecting char budget."""
    total_chars = 0
    selected = []
    blocks = []
    seen = set()

    for c in ranked_chunks:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        header = f"[{cid} p.{c['page']}] "
        block = header + c["text"]
        if total_chars + len(block) > budget:
            continue
        total_chars += len(block)
        selected.append(c)
        blocks.append(block)

    return "\n\n".join(blocks), selected


# --- Lexical Retriever (BM25 / TF-IDF) ---

class LexicalRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.N = len(chunks)

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return self.chunks[:top_k]

        k1 = 1.5
        b = 0.75
        scores = []

        for i, c in enumerate(self.chunks):
            # Apply chapter / metadata filter if provided
            if filters:
                match_filter = True
                for k, v in filters.items():
                    meta = c.get("metadata", {})
                    if meta.get(k) != v and c.get(k) != v:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            tf = Counter(self.doc_tokens[i])
            doc_len = self.doc_lens[i]
            score = 0.0

            for tok in q_tokens:
                if tok not in tf:
                    continue
                freq = tf[tok]
                df_val = self.df.get(tok, 0)
                idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)
                denom = freq + k1 * (1.0 - b + b * (doc_len / max(self.avg_dl, 1.0)))
                score += idf * ((freq * (k1 + 1.0)) / denom)

            scores.append((score, i, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, _, c in scores[:top_k]]


# --- Dense Retriever (Gemini / SentenceTransformer + Cosine) ---

class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str = None):
        self.chunks = chunks
        self.model_name = model_name
        self._init_model()
        self._embed_corpus()

    def _init_model(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.is_gemini = False
        if gemini_key and len(gemini_key) > 10:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=gemini_key)
                self._types = types
                self.embed_model = self.model_name or "models/gemini-embedding-001"
                self.dimensions = 768
                self.is_gemini = True
                print(f"[DenseRetriever] Successfully configured Gemini API with {self.embed_model} (dim={self.dimensions})")
            except Exception as e:
                print(f"[DenseRetriever] Could not initialize Gemini API: {e}")
                self.is_gemini = False

        if not self.is_gemini:
            from sentence_transformers import SentenceTransformer
            self.embed_model = self.model_name or "all-MiniLM-L6-v2"
            self.local_model = SentenceTransformer(self.embed_model)
            self.dimensions = self.local_model.get_embedding_dimension()
            print(f"[DenseRetriever] Fallback to local model: {self.embed_model} (dim={self.dimensions})")

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if not self.is_gemini:
            return self.local_model.encode(texts, normalize_embeddings=True).tolist()

        # Batch embedding via Gemini API matching Day 6
        all_embeddings: list[list[float]] = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.embed_model,
                        contents=batch,
                        config=self._types.EmbedContentConfig(output_dimensionality=self.dimensions)
                    )
                    batch_vectors = [emb.values for emb in response.embeddings]
                    all_embeddings.extend(batch_vectors)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_sec = 30
                        print(f"[DenseRetriever] Gemini 429 quota. Waiting {wait_sec}s...")
                        time.sleep(wait_sec)
                    elif attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"[DenseRetriever] Gemini API failed after {max_retries} attempts ({e}). Falling back to local SentenceTransformer...")
                        from sentence_transformers import SentenceTransformer
                        if not hasattr(self, "local_model"):
                            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
                        self.is_gemini = False
                        return self.local_model.encode(texts, normalize_embeddings=True).tolist()

        return all_embeddings

    def _embed_corpus(self):
        print(f"[DenseRetriever] Generating embeddings for {len(self.chunks)} chunks using {self.embed_model}...")
        texts = [c["text"] for c in self.chunks]
        self.corpus_vectors = self._embed_texts(texts)

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
        q_vec = self._embed_texts([query])[0]
        # Dot product for normalized vectors = Cosine similarity
        import numpy as np
        q_arr = np.array(q_vec)
        norm_q = np.linalg.norm(q_arr)
        if norm_q > 0:
            q_arr = q_arr / norm_q

        scored_chunks = []
        for i, c in enumerate(self.chunks):
            if filters:
                match_filter = True
                for k, v in filters.items():
                    meta = c.get("metadata", {})
                    if meta.get(k) != v and c.get(k) != v:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            v_arr = np.array(self.corpus_vectors[i])
            norm_v = np.linalg.norm(v_arr)
            sim = float(np.dot(q_arr, v_arr) / (norm_v if norm_v > 0 else 1.0))
            scored_chunks.append((sim, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks[:top_k]]


# --- Hybrid Retriever (RRF: Reciprocal Rank Fusion) ---

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, lexical_retriever: LexicalRetriever):
        self.dense = dense_retriever
        self.lexical = lexical_retriever

    def search(self, query: str, top_k: int = 10, filters: dict = None, rrf_k: int = 60) -> list[dict]:
        dense_hits = self.dense.search(query, top_k=top_k * 2, filters=filters)
        lexical_hits = self.lexical.search(query, top_k=top_k * 2, filters=filters)

        scores = Counter()
        chunk_map = {}

        for rank, c in enumerate(dense_hits):
            cid = c["id"]
            chunk_map[cid] = c
            scores[cid] += 1.0 / (rrf_k + rank + 1)

        for rank, c in enumerate(lexical_hits):
            cid = c["id"]
            chunk_map[cid] = c
            scores[cid] += 1.0 / (rrf_k + rank + 1)

        ranked = scores.most_common(top_k)
        return [chunk_map[cid] for cid, _ in ranked]


# --- Benchmark Evaluator ---

def evaluate_retrieval(retriever, questions: list[dict], top_k: int = 3, budget: int = 800) -> tuple[dict, list[dict]]:
    traces = []
    latencies = []
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    total_context_chars = 0

    for q in questions:
        t0 = time.perf_counter()
        query_text = q["question"]
        filters = q.get("filter")
        hits = retriever.search(query_text, top_k=max(top_k, 3), filters=filters)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        gold = q["gold"]
        h1 = hit(hits[0], gold) if hits else False
        h3 = any(hit(c, gold) for c in hits[:3])
        _, ctx_chunks = build_context(hits, budget=budget)
        ctx_hit = any(hit(c, gold) for c in ctx_chunks)

        if h1: hit1_count += 1
        if h3: hit3_count += 1
        if ctx_hit: context_hit_count += 1
        total_context_chars += sum(len(c["text"]) for c in ctx_chunks)

        traces.append({
            "id": q["id"],
            "question": q["question"],
            "top_hits": [{"id": c["id"], "page": c["page"], "text": c["text"][:100]} for c in hits[:top_k]],
            "hit1": h1,
            "hit3": h3,
            "context_hit": ctx_hit
        })

    n = len(questions)
    summary = {
        "n": n,
        "hit1": hit1_count,
        "hit1_rate": round(hit1_count / n * 100, 2) if n else 0,
        "hit3": hit3_count,
        "hit3_rate": round(hit3_count / n * 100, 2) if n else 0,
        "context_hit": context_hit_count,
        "context_evidence_rate": round(context_hit_count / n * 100, 2) if n else 0,
        "avg_context_chars": round(total_context_chars / n, 1) if n else 0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0
    }
    return summary, traces
