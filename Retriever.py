"""Retriever strategies + factory. Each implements .fit(chunk_texts, chunk_vecs)
and .query(query_texts, query_vecs, k) -> np.ndarray of shape (n_queries, k).
Splitting fit/query means an index is built once per (splitter, config) and
reused across every retriever call at MAX_K, then sliced per k downstream."""

import numpy as np


class CosineNumpyRetriever:
    """Exact cosine similarity via a plain matrix multiply. Assumes normalized vectors."""
    def fit(self, chunk_texts, chunk_vecs):
        self._chunk_vecs = chunk_vecs

    def query(self, query_texts, query_vecs, k):
        sims = query_vecs @ self._chunk_vecs.T
        k = min(k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row_scores = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        return np.take_along_axis(idx, order, axis=1)


class FaissFlatRetriever:
    """Exact search via FAISS IndexFlatIP. Inner product == cosine on normalized vectors."""
    def fit(self, chunk_texts, chunk_vecs):
        import faiss
        self._index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        self._index.add(np.ascontiguousarray(chunk_vecs.astype("float32")))

    def query(self, query_texts, query_vecs, k):
        _, idx = self._index.search(np.ascontiguousarray(query_vecs.astype("float32")), k)
        return idx


class FaissHNSWRetriever:
    """Approximate nearest-neighbor search — faster at scale, small recall cost."""
    def fit(self, chunk_texts, chunk_vecs):
        import faiss
        self._index = faiss.IndexHNSWFlat(chunk_vecs.shape[1], 32)
        self._index.hnsw.efConstruction = 40
        self._index.add(np.ascontiguousarray(chunk_vecs.astype("float32")))
        self._index.hnsw.efSearch = 64

    def query(self, query_texts, query_vecs, k):
        _, idx = self._index.search(np.ascontiguousarray(query_vecs.astype("float32")), k)
        return idx


class ChromaRetriever:
    """Wraps an ephemeral (in-memory) ChromaDB collection — useful for comparing
    against a production Chroma-backed pipeline."""
    def fit(self, chunk_texts, chunk_vecs):
        import chromadb
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(name="finrag_eval")
        ids = [str(i) for i in range(len(chunk_texts))]
        batch = 500
        for start in range(0, len(ids), batch):
            self._collection.add(
                ids=ids[start:start + batch],
                embeddings=chunk_vecs[start:start + batch].tolist(),
                documents=chunk_texts[start:start + batch],
            )

    def query(self, query_texts, query_vecs, k):
        result = self._collection.query(query_embeddings=query_vecs.tolist(), n_results=k)
        return np.array([[int(i) for i in row] for row in result["ids"]])


class BM25Retriever:
    """Sparse keyword baseline — no embeddings involved at all."""
    def fit(self, chunk_texts, chunk_vecs):
        from rank_bm25 import BM25Okapi
        self._chunk_texts = chunk_texts
        self._bm25 = BM25Okapi([t.lower().split() for t in chunk_texts])

    def query(self, query_texts, query_vecs, k):
        rows = []
        for q in query_texts:
            scores = self._bm25.get_scores(q.lower().split())
            rows.append(np.argsort(scores)[::-1][:k])
        return np.array(rows)


class HybridRRFRetriever:
    """
    Fuses a dense retriever and BM25 using Reciprocal Rank Fusion — no
    learned weighting needed. score(doc) = sum over rankers of
    1 / (rrf_k + rank_in_that_ranker). Financial text mixes exact-match
    terms (tickers, line-item names) with paraphrase-sensitive queries,
    so this is a real hypothesis about FinDER specifically, not just
    "combine everything" for its own sake.

    candidate_pool controls how many results each sub-retriever contributes
    to the fusion before truncating to k — must be >= k, and larger pools
    give RRF more to work with at the cost of more BM25/dense query time.
    """
    def __init__(self, dense_retriever_name: str = "cosine_numpy",
                 rrf_k: int = 60, candidate_pool: int = 100):
        self._dense = get_retriever(dense_retriever_name)
        self._bm25 = BM25Retriever()
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool

    def fit(self, chunk_texts, chunk_vecs):
        self._dense.fit(chunk_texts, chunk_vecs)
        self._bm25.fit(chunk_texts, chunk_vecs)

    def query(self, query_texts, query_vecs, k):
        pool = max(k, self._candidate_pool)
        dense_idx = self._dense.query(query_texts, query_vecs, pool)
        bm25_idx = self._bm25.query(query_texts, query_vecs, pool)

        results = []
        for qi in range(len(query_texts)):
            scores: dict[int, float] = {}
            for rank, idx in enumerate(dense_idx[qi]):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (self._rrf_k + rank + 1)
            for rank, idx in enumerate(bm25_idx[qi]):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (self._rrf_k + rank + 1)
            top = sorted(scores.items(), key=lambda item: -item[1])[:k]
            results.append([idx for idx, _ in top])
        return np.array(results)


RETRIEVER_REGISTRY = {
    "cosine_numpy": CosineNumpyRetriever,
    "faiss_flat": FaissFlatRetriever,
    "faiss_hnsw": FaissHNSWRetriever,
    "chroma": ChromaRetriever,
    "bm25": BM25Retriever,
    "hybrid_rrf": HybridRRFRetriever,
}


def get_retriever(name: str):
    """Factory: construct a fresh retriever instance by name.
    Fresh instance per call matters — retrievers hold fit-time state (an index),
    and reusing one across sweep configs would leak chunks from a previous config."""
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retriever '{name}'. Registered: {list(RETRIEVER_REGISTRY)}")
    return RETRIEVER_REGISTRY[name]()
    