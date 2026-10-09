"""
Earnings-call transcript RAG (Retrieval-Augmented Generation)
=============================================================

Pipeline:
  1. FETCH     - download a transcript from Alpha Vantage (cached to disk)
  2. CHUNK     - split it into speaker-aware pieces with metadata
  3. STORE     - put chunks in a vector store you choose (--db):
                   milvus   - Milvus Lite local file (DEFAULT), hybrid dense + built-in BM25
                   qdrant   - Qdrant local folder, hybrid dense + sparse (server-side IDF)
                   pinecone - Pinecone serverless cloud, dense + metadata filters
                   memory   - in-process BM25 + embeddings, rebuilt every run (no database)
  4. RETRIEVE  - hybrid search with RRF fusion + metadata filters, optional cross-encoder rerank
  5. GENERATE  - ask an LLM to answer ONLY from retrieved excerpts, with citations

Install:
  pip install requests rank_bm25 numpy sentence-transformers anthropic
  pip install pymilvus milvus-lite      # --db milvus   (default)
  pip install qdrant-client             # --db qdrant
  pip install pinecone                  # --db pinecone

Usage:
  # 1) Store transcripts once (persisted in the database)
  python earnings_rag.py ingest IBM:2026Q2 IBM:2026Q1                  # Milvus by default
  python earnings_rag.py --db qdrant ingest IBM:2026Q2 IBM:2026Q1
  # 2) Ask questions any time later
  python earnings_rag.py ask "Is the software demand deferred or lost?" --symbol IBM --quarter 2026Q2
  python earnings_rag.py ask "..." --section qa --no-llm               # retrieval only, no API call
  # In-memory mode has no database, so pass the transcripts with the question
  python earnings_rag.py --db memory ask "..." --pairs IBM:2026Q2

Environment variables:
  ALPHAVANTAGE_API_KEY  - required for fetching transcripts
  ANTHROPIC_API_KEY     - required for answer generation (not for --no-llm)
  PINECONE_API_KEY      - only for --db pinecone
  VECTOR_DB             - default store if --db is not given (milvus if unset)
  LLM_MODEL             - optional, overrides the default model
  EMBED_MODEL           - optional, overrides the embedding model
  RERANK_MODEL          - optional, overrides the reranker
  TRANSCRIPT_CACHE      - optional, folder for cached transcripts
  MILVUS_URI            - optional, Milvus Lite file or server URL (default ./milvus_earnings.db)
  QDRANT_PATH / QDRANT_URL - optional, Qdrant local folder or server URL
"""
from __future__ import annotations   # allows "list[dict]" style type hints on older Python
import argparse, json, os, re, pathlib, uuid, zlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np
import requests
from rank_bm25 import BM25Okapi       # classic keyword-ranking algorithm (like a search engine)

# ---------------------------------------------------------------- Config
# All settings can be overridden with environment variables, so you don't edit code to switch models.
CACHE = pathlib.Path(os.getenv("TRANSCRIPT_CACHE", "./transcripts"))  # where raw transcripts are saved
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")                  # model that writes the answer
# Small, fast, good-quality embedding model (384-dim vectors). Runs on CPU.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Cross-encoder reads (query, passage) together -> more accurate relevance than embeddings alone,
# but slower, so we only apply it to the top candidates.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Which store to use when --db is not given. Milvus is the default.
VECTOR_DB = os.getenv("VECTOR_DB", "milvus")
DB_CHOICES = ("milvus", "qdrant", "pinecone", "memory")
COLLECTION = "earnings_calls"          # collection / index name in every database
# Metadata fields we filter on. Every database gets an index on these.
FILTER_FIELDS = ("symbol", "quarter", "speaker", "section")


# ================================================================ 1. FETCH
def fetch_transcript(symbol: str, quarter: str) -> list[dict]:
    """
    Download one earnings-call transcript from Alpha Vantage.

    Args:
        symbol:  ticker, e.g. "IBM"
        quarter: fiscal quarter as "YYYYQn", e.g. "2026Q2"

    Returns:
        A list of speaker turns, each a dict:
        {"speaker": ..., "title": ..., "content": ..., "sentiment": "0.7"}

    The result is cached as JSON on disk so repeat runs don't burn API quota
    (Alpha Vantage free/low tiers are rate-limited).
    """
    CACHE.mkdir(exist_ok=True)                           # create cache folder if missing
    fp = CACHE / f"{symbol.upper()}_{quarter}.json"      # e.g. ./transcripts/IBM_2026Q2.json
    if fp.exists():                                      # cache hit -> skip the network call
        return json.loads(fp.read_text())

    # Alpha Vantage uses one endpoint; the "function" param selects the dataset.
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": symbol,
                "quarter": quarter, "apikey": os.environ["ALPHAVANTAGE_API_KEY"]},
        timeout=60,                                      # don't hang forever on a slow response
    )
    r.raise_for_status()                                 # catches real HTTP errors (4xx/5xx)
    data = r.json()                                      # parse JSON body into a Python dict

    # On success data looks like {"symbol":..., "quarter":..., "transcript":[...]}.
    # .get() returns None instead of crashing if "transcript" is absent.
    turns = data.get("transcript")

    # Alpha Vantage often returns HTTP 200 even on failure, with a body like
    # {"Information": "rate limit..."} or {"Error Message": "..."}. `not turns` is True
    # for both None (key missing) and [] (no transcript), so we stop with the real message.
    if not turns:
        raise RuntimeError(f"No transcript for {symbol} {quarter}: {data}")

    fp.write_text(json.dumps(turns))                     # save for next time
    return turns


# ================================================================ 2. CHUNK
@dataclass
class Chunk:
    """One retrievable unit of text plus the metadata used for filtering and citations."""
    text: str             # the passage itself
    symbol: str           # ticker, e.g. "IBM"
    quarter: str          # e.g. "2026Q2"
    speaker: str          # e.g. "Arvind Krishna"
    title: str            # role, e.g. "Chief Executive Officer" or "Analyst"
    section: str          # "prepared" (scripted remarks) or "qa" (analyst Q&A)
    sentiment: float      # Alpha Vantage's per-turn sentiment score
    turn_idx: int         # position of the turn in the original transcript
    id: str = field(default="")   # unique id used for citations, set after creation


def chunk_transcript(turns, symbol, quarter, max_words=300, overlap=50) -> list[Chunk]:
    """
    Convert raw speaker turns into Chunks.

    Strategy:
      - Respect speaker boundaries: a chunk never mixes two speakers, so every
        citation can be attributed to one person and role.
      - Long turns (e.g. a 2,000-word CEO opening) are split into windows of
        `max_words`, overlapping by `overlap` words so a sentence cut at a boundary
        still appears whole in at least one chunk.
      - Tag each chunk as "prepared" or "qa" so you can filter, e.g. "only what
        analysts asked" or "only scripted guidance".
      - Drop Operator turns ("Welcome, you may disconnect...") - pure noise for retrieval.
    """
    chunks, section = [], "prepared"         # every call starts with prepared remarks
    for i, t in enumerate(turns):
        title = (t.get("title") or "").strip()

        # The first time an analyst speaks, Q&A has started. Every later turn
        # (including management answers) is tagged "qa".
        if "analyst" in title.lower():
            section = "qa"

        if t.get("speaker") == "Operator":   # skip call-logistics boilerplate
            continue

        words = t["content"].split()
        step = max_words - overlap           # how far the window advances each time (300-50 = 250)

        # Slide a window over the turn. max(len, 1) guarantees at least one iteration.
        for j, start in enumerate(range(0, max(len(words), 1), step)):
            piece = " ".join(words[start:start + max_words])
            if not piece:                    # empty turn -> nothing to index
                break
            c = Chunk(piece, symbol, quarter, t["speaker"], title, section,
                      float(t.get("sentiment") or 0), i)   # sentiment arrives as a string
            # Stable, readable id: <ticker>-<quarter>-t<turn>-<window>, e.g. IBM-2026Q2-t3-0
            c.id = f"{symbol}-{quarter}-t{i}-{j}"
            chunks.append(c)
            if start + max_words >= len(words):   # this window reached the end of the turn
                break
    return chunks


# ================================================================ 3. STORE
# Shared helpers used by every store.
def chunk_text(c: Chunk) -> str:
    """Text that gets embedded/indexed. The header makes company, speaker and role searchable."""
    return f"{c.symbol} {c.quarter} {c.speaker} ({c.title}): {c.text}"


def as_list(v):
    """Filter values can be one value or a list (a list means OR)."""
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


class Embedder:
    """
    Dense embeddings via sentence-transformers. Vectors are L2-normalized, so cosine
    similarity == dot product in every backend. Loaded once and shared by store + queries.
    """
    def __init__(self, model: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        # 384 for bge-small. Newer sentence-transformers renamed the method; support both.
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False)


# ---------------------------------------------------------------- 3a. In-memory store
# Tokenizer for BM25: lowercase and keep letters, digits, "$", "%" and "." so that
# figures like "$24.6", "18%" and model names like "z17" survive as single tokens.
_tok = re.compile(r"[a-z0-9$%.]+")
def tokenize(s: str) -> list[str]:
    return _tok.findall(s.lower())


class HybridIndex:
    """
    In-memory hybrid search over Chunks.

    Why hybrid?
      - BM25 (lexical) is precise for exact terms: numbers, product names, tickers.
      - Dense embeddings (semantic) catch paraphrases: a query about "demand being lost"
        can match a passage saying "deferral, not destruction".
      Each covers the other's blind spot; fusing them beats either alone.

    Rebuilt from scratch every run (nothing persisted). Fine for tens to a few hundred
    transcripts; beyond that use one of the database stores below (--db milvus/qdrant/pinecone).
    """
    def __init__(self, chunks: list[Chunk], use_dense=True, use_rerank=True):
        self.chunks = chunks
        # Prefix each passage with company/quarter/speaker/role so those words are
        # searchable too (e.g. a query mentioning "CFO" or "2026Q2" can match lexically).
        self.docs = [chunk_text(c) for c in chunks]

        # BM25 index over tokenized passages (built instantly, no model needed).
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

        self.encoder = self.reranker = None
        if use_dense:
            # Imported lazily so BM25-only mode works without sentence-transformers installed.
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(EMBED_MODEL)
            # normalize_embeddings=True -> unit vectors, so a dot product == cosine similarity.
            self.emb = self.encoder.encode(self.docs, normalize_embeddings=True,
                                           batch_size=32, show_progress_bar=False)
        if use_rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)

    def _mask(self, filters: dict | None) -> np.ndarray:
        """
        Build a boolean array (one entry per chunk) that is True where the chunk
        matches ALL filters. Each filter value can be a single value or a list (OR).
        Example: {"symbol": "IBM", "quarter": ["2026Q1", "2026Q2"], "section": "qa"}
        """
        m = np.ones(len(self.chunks), bool)               # start with "everything allowed"
        for k, v in (filters or {}).items():
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            m &= np.array([getattr(c, k) in vals for c in self.chunks])   # AND across filters
        return m

    def search(self, query: str, k=8, filters: dict | None = None,
               candidates=40, rrf_k=60) -> list[tuple[Chunk, float]]:
        """
        Return the top-k (Chunk, score) pairs for a query.

        Args:
            k:          number of results returned to the LLM
            filters:    metadata filters, see _mask()
            candidates: how many results each retriever contributes before fusion
            rrf_k:      RRF smoothing constant; 60 is the standard value from the original paper
        """
        mask = self._mask(filters)
        if not mask.any():                                 # filters excluded everything
            return []

        # --- Lexical retrieval: score every chunk, push filtered-out chunks to -inf.
        bm = np.where(mask, self.bm25.get_scores(tokenize(query)), -np.inf)
        ranks = [np.argsort(-bm)[:candidates]]             # chunk indices, best first

        # --- Semantic retrieval: cosine similarity between query and every chunk.
        if self.encoder is not None:
            q = self.encoder.encode([query], normalize_embeddings=True)[0]
            ds = np.where(mask, self.emb @ q, -np.inf)     # matrix @ vector = all similarities at once
            ranks.append(np.argsort(-ds)[:candidates])

        # --- Reciprocal Rank Fusion (RRF):
        # score(chunk) = sum over retrievers of 1 / (rrf_k + rank). It uses ranks, not raw
        # scores, so BM25 scores and cosine similarities (different scales) combine fairly.
        # A chunk ranked high by BOTH retrievers ends up on top.
        fused: dict[int, float] = {}
        for r in ranks:
            for pos, idx in enumerate(r):
                if mask[idx]:                              # ignore filtered-out padding
                    fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + pos + 1)
        cand = sorted(fused, key=fused.get, reverse=True)[:candidates]

        # --- Optional rerank: the cross-encoder scores each (query, passage) pair jointly,
        # which is slower but more accurate, so it only runs on the fused shortlist.
        if self.reranker is not None and cand:
            scores = self.reranker.predict([(query, self.docs[i]) for i in cand])
            cand = [cand[i] for i in np.argsort(-scores)]  # reorder by reranker score
            return [(self.chunks[i], float(s)) for i, s in zip(cand[:k], sorted(scores, reverse=True))]

        return [(self.chunks[i], fused[i]) for i in cand[:k]]


# ---------------------------------------------------------------- 3b. Database stores
class VectorStore(ABC):
    """Common interface. Each subclass translates it into its database's API."""

    @abstractmethod
    def create(self, dim: int, recreate: bool = False) -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: str, query_vec: np.ndarray, k: int = 8,
               filters: dict | None = None) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    _as_list = staticmethod(as_list)


# ---------------------------------------------------------------- 3c. Qdrant
# Sparse "BM25-style" vectors for Qdrant. Each token maps to a dimension by a STABLE hash
# (crc32, not Python's hash(), which is randomized per process and would corrupt the
# index across runs). Values are raw term frequencies; Qdrant applies IDF server-side
# (Modifier.IDF), so TF on our side x IDF on Qdrant's side approximates BM25.
def sparse_encode(text: str) -> tuple[list[int], list[float]]:
    tf = Counter(tokenize(text))                     # same tokenizer as the in-memory BM25
    idx = {}
    for tok, n in tf.items():
        h = zlib.crc32(tok.encode()) & 0x7FFFFFFF      # non-negative 31-bit index
        idx[h] = idx.get(h, 0) + n                     # merge (rare) hash collisions
    return list(idx.keys()), [float(v) for v in idx.values()]


class QdrantStore(VectorStore):
    """
    Qdrant: each point has two named vectors:
      "dense"  - embedding, cosine distance, HNSW index
      "sparse" - term frequencies with server-side IDF (lexical match on numbers/names)
    Search runs both as "prefetch" queries and fuses them server-side with RRF.
    """
    def __init__(self, path: str | None = os.getenv("QDRANT_PATH", "./qdrant_data"),
                 url: str | None = os.getenv("QDRANT_URL"),
                 collection: str = COLLECTION):
        from qdrant_client import QdrantClient
        # Local mode: data lives in a folder, no server needed. Use url= for Docker/Cloud.
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        self.name = collection

    def create(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models as m
        if self.client.collection_exists(self.name):
            if not recreate:
                return
            self.client.delete_collection(self.name)
        self.client.create_collection(
            self.name,
            vectors_config={"dense": m.VectorParams(size=dim, distance=m.Distance.COSINE)},
            sparse_vectors_config={"sparse": m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        # Payload indexes make filtered search fast (ignored in local mode, used on a server).
        for f in FILTER_FIELDS:
            self.client.create_payload_index(self.name, f, m.PayloadSchemaType.KEYWORD)

    def upsert(self, chunks, vectors) -> None:
        from qdrant_client import models as m
        points = []
        for c, v in zip(chunks, vectors):
            si, sv = sparse_encode(chunk_text(c))
            points.append(m.PointStruct(
                # Qdrant ids must be ints or UUIDs. uuid5 turns our readable id into a
                # deterministic UUID, so re-ingesting the same chunk overwrites it (idempotent).
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector={"dense": v.tolist(), "sparse": m.SparseVector(indices=si, values=sv)},
                payload=asdict(c),
            ))
        self.client.upsert(self.name, points=points, wait=True)

    def _filter(self, filters):
        """{"symbol": "IBM", "quarter": [...]} -> Qdrant Filter (AND of fields, OR within a list)."""
        from qdrant_client import models as m
        if not filters:
            return None
        conds = []
        for k, v in filters.items():
            vals = self._as_list(v)
            match = m.MatchValue(value=vals[0]) if len(vals) == 1 else m.MatchAny(any=vals)
            conds.append(m.FieldCondition(key=k, match=match))
        return m.Filter(must=conds)

    def search(self, query, query_vec, k=8, filters=None):
        from qdrant_client import models as m
        flt = self._filter(filters)
        si, sv = sparse_encode(query)
        res = self.client.query_points(
            self.name,
            prefetch=[   # two candidate lists, each already filtered
                m.Prefetch(query=query_vec.tolist(), using="dense", filter=flt, limit=40),
                m.Prefetch(query=m.SparseVector(indices=si, values=sv), using="sparse",
                           filter=flt, limit=40),
            ],
            query=m.FusionQuery(fusion=m.Fusion.RRF),   # fuse them with RRF, server-side
            limit=k, with_payload=True,
        )
        return [(Chunk(**p.payload), float(p.score)) for p in res.points]

    def count(self):
        return self.client.count(self.name, exact=True).count


# ---------------------------------------------------------------- 3d. Milvus (default)
class MilvusStore(VectorStore):
    """
    Milvus: explicit schema. The "text" field feeds a built-in BM25 Function that fills a
    sparse vector field automatically, so Milvus does tokenizing and IDF itself.
    Search runs a dense request and a BM25 request, then fuses them with RRFRanker.
    Milvus Lite (a local .db file) is for learning/prototyping, not production.
    """
    def __init__(self, uri: str = os.getenv("MILVUS_URI", "./milvus_earnings.db"),
                 collection: str = COLLECTION):
        from pymilvus import MilvusClient
        # A file path -> Milvus Lite (embedded). "http://host:19530" -> a Milvus server.
        self.client = MilvusClient(uri)
        self.name = collection
        self._loaded = False

    def _ensure_loaded(self):
        """
        Milvus only searches collections that are "loaded" into memory, and load state is
        per process: a collection saved by `ingest` starts out "released" in the next process
        (`ask`). Load it once before the first search/count in this process.
        """
        if not self._loaded:
            self.client.load_collection(self.name)
            self._loaded = True

    def create(self, dim, recreate=False):
        from pymilvus import DataType, Function, FunctionType
        if self.client.has_collection(self.name):
            if not recreate:
                return
            self.client.drop_collection(self.name)
            self._loaded = False

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)   # filled by BM25 Function
        for f in FILTER_FIELDS + ("title",):
            schema.add_field(f, DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)  # raw passage
        schema.add_field("sentiment", DataType.FLOAT)
        schema.add_field("turn_idx", DataType.INT64)
        # text -> sparse BM25 vector, computed by Milvus on insert and on query
        schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                     input_field_names=["text"], output_field_names=["sparse"]))

        idx = self.client.prepare_index_params()
        idx.add_index("dense", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 200})
        idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self.client.create_collection(self.name, schema=schema, index_params=idx)

    def upsert(self, chunks, vectors):
        rows = [{
            "id": c.id, "text": chunk_text(c), "dense": v.tolist(),
            "symbol": c.symbol, "quarter": c.quarter, "speaker": c.speaker,
            "section": c.section, "title": c.title, "chunk_text": c.text,
            "sentiment": c.sentiment, "turn_idx": c.turn_idx,
        } for c, v in zip(chunks, vectors)]   # no "sparse": the BM25 Function generates it
        self.client.upsert(self.name, rows)

    def _filter(self, filters) -> str:
        """{"symbol": "IBM", "quarter": [...]} -> 'symbol == "IBM" and quarter in ["..",".."]'"""
        parts = []
        for k, v in (filters or {}).items():
            vals = self._as_list(v)
            q = ", ".join(f'"{x}"' for x in vals)
            parts.append(f'{k} == "{vals[0]}"' if len(vals) == 1 else f"{k} in [{q}]")
        return " and ".join(parts)

    _OUT = ["chunk_text", "symbol", "quarter", "speaker", "title", "section",
            "sentiment", "turn_idx"]

    def search(self, query, query_vec, k=8, filters=None):
        from pymilvus import AnnSearchRequest, RRFRanker
        self._ensure_loaded()
        expr = self._filter(filters)
        dense_req = AnnSearchRequest(data=[query_vec.tolist()], anns_field="dense",
                                     param={"metric_type": "COSINE"}, limit=40, expr=expr)
        bm25_req = AnnSearchRequest(data=[query], anns_field="sparse",    # raw text in;
                                    param={"metric_type": "BM25"}, limit=40, expr=expr)  # Milvus tokenizes
        hits = self.client.hybrid_search(self.name, [dense_req, bm25_req], ranker=RRFRanker(60),
                                         limit=k, output_fields=self._OUT)[0]
        out = []
        for h in hits:
            e = h["entity"]
            c = Chunk(e["chunk_text"], e["symbol"], e["quarter"], e["speaker"], e["title"],
                      e["section"], float(e["sentiment"]), int(e["turn_idx"]), id=h["id"])
            out.append((c, float(h["distance"])))
        return out

    def count(self):
        self._ensure_loaded()
        res = self.client.query(self.name, filter="", output_fields=["count(*)"])
        return int(res[0]["count(*)"])


# ---------------------------------------------------------------- 3e. Pinecone
class PineconeStore(VectorStore):
    """
    Pinecone: fully managed serverless index; no schema beyond dimension + metric.
    Metadata is schemaless JSON, filtered with a MongoDB-style syntax.
    Implemented here as dense-only. Pinecone hybrid needs either a separate sparse index
    or a dotproduct index with sparse_values, which is a good follow-up exercise.
    Needs PINECONE_API_KEY. The Starter (free) plan covers this size of data.
    """
    def __init__(self, index_name: str = "earnings-calls", namespace: str = "transcripts",
                 cloud: str = "aws", region: str = "us-east-1"):
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name, self.ns = index_name, namespace
        self.cloud, self.region = cloud, region
        self.index = None

    def create(self, dim, recreate=False):
        from pinecone import ServerlessSpec
        if self.pc.has_index(self.index_name):
            if recreate:
                self.pc.delete_index(self.index_name)
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(self.index_name, dimension=dim, metric="cosine",
                                 spec=ServerlessSpec(cloud=self.cloud, region=self.region))
        self.index = self.pc.Index(self.index_name)

    def upsert(self, chunks, vectors):
        # Pinecone metadata values must be str/number/bool/list[str]; Chunk fields all qualify.
        # Metadata is capped at 40 KB per record, so 300-word chunks fit easily.
        self.index.upsert(vectors=[{"id": c.id, "values": v.tolist(), "metadata": asdict(c)}
                                   for c, v in zip(chunks, vectors)],
                          namespace=self.ns, batch_size=100)

    def _filter(self, filters):
        """{"symbol": "IBM", "quarter": [...]} -> {"symbol": {"$eq": "IBM"}, "quarter": {"$in": [...]}}"""
        out = {}
        for k, v in (filters or {}).items():
            vals = self._as_list(v)
            out[k] = {"$eq": vals[0]} if len(vals) == 1 else {"$in": vals}
        return out or None

    def search(self, query, query_vec, k=8, filters=None):
        res = self.index.query(vector=query_vec.tolist(), top_k=k, namespace=self.ns,
                               filter=self._filter(filters), include_metadata=True)
        return [(Chunk(**m.metadata), float(m.score)) for m in res.matches]

    def count(self):
        stats = self.index.describe_index_stats()
        return stats.namespaces.get(self.ns).vector_count if self.ns in stats.namespaces else 0



# ---------------------------------------------------------------- 3f. Selection + retrieval
def make_store(db: str) -> VectorStore:
    """Map a --db name to a store class. 'memory' is handled separately (no persistence)."""
    stores = {"milvus": MilvusStore, "qdrant": QdrantStore, "pinecone": PineconeStore}
    if db not in stores:
        raise ValueError(f"Unknown --db {db!r}; choose one of {DB_CHOICES}")
    return stores[db]()


def ingest(store: VectorStore, embedder, pairs: list[tuple[str, str]], recreate=False) -> int:
    """Fetch -> chunk -> embed -> upsert. Safe to re-run: chunk ids are deterministic."""
    store.create(embedder.dim, recreate=recreate)
    total = 0
    for sym, q in pairs:
        chunks = chunk_transcript(fetch_transcript(sym, q), sym, q)
        store.upsert(chunks, embedder.embed([chunk_text(c) for c in chunks]))
        total += len(chunks)
        print(f"  {sym} {q}: {len(chunks)} chunks")
    return total


class StoreRetriever:
    """
    Wraps any database store with the same .search(query, k, filters) signature as
    HybridIndex, so answer() works the same whichever store was chosen.
    Optionally reranks the database's results with the cross-encoder.
    """
    def __init__(self, store: VectorStore, embedder, use_rerank=True, candidates=20):
        self.store, self.embedder, self.candidates = store, embedder, candidates
        self.reranker = None
        if use_rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)

    def search(self, query, k=8, filters=None):
        qv = self.embedder.embed([query])[0]
        # Fetch a wider shortlist when reranking, then keep the best k.
        n = max(k, self.candidates) if self.reranker else k
        hits = self.store.search(query, qv, k=n, filters=filters)
        if self.reranker is None or not hits:
            return hits
        scores = self.reranker.predict([(query, chunk_text(c)) for c, _ in hits])
        order = np.argsort(-scores)[:k]
        return [(hits[i][0], float(scores[i])) for i in order]


# ================================================================ 4. GENERATE
# System prompt = the grounding contract. It is what keeps the LLM from mixing in
# outside knowledge or inventing figures.
SYSTEM = """You answer questions about company earnings calls using ONLY the excerpts provided.
Rules:
- Cite every factual claim with the excerpt id in square brackets, e.g. [IBM-2026Q2-t3-0].
- Distinguish management statements from analyst questions; name the speaker and role.
- Quote numbers exactly as stated; do not compute or infer figures that are not in the excerpts.
- Flag forward-looking statements (guidance, expectations) as such.
- If the excerpts do not answer the question, say so explicitly. Do not use outside knowledge."""


def format_context(hits) -> str:
    """
    Wrap each retrieved chunk in an XML-style tag carrying its metadata.
    Tags make excerpt boundaries unambiguous for the LLM and give it the id to cite.
    """
    return "\n\n".join(
        f"<excerpt id=\"{c.id}\" company=\"{c.symbol}\" quarter=\"{c.quarter}\" "
        f"speaker=\"{c.speaker}\" role=\"{c.title}\" section=\"{c.section}\" "
        f"sentiment=\"{c.sentiment}\">\n{c.text}\n</excerpt>" for c, _ in hits)


def answer(index, question: str, k=8, filters=None, max_tokens=1200) -> dict:
    """
    Retrieve relevant excerpts, ask the LLM, and audit its citations.

    Returns a dict:
        answer           - the LLM's text answer
        sources          - the retrieved chunks (with scores) that were sent to the LLM
        uncited_ids      - retrieved but unused chunks (high count = noisy retrieval)
        hallucinated_ids - ids the LLM cited that were NOT provided (should always be empty)

    `index` is anything with .search(query, k, filters): a HybridIndex (--db memory)
    or a StoreRetriever over Milvus / Qdrant / Pinecone.
    To use another LLM provider, replace only the block marked below.
    """
    hits = index.search(question, k=k, filters=filters)
    if not hits:
        return {"answer": "No matching excerpts for these filters.", "sources": []}

    # ---- LLM call (swap this block for OpenAI / Gemini / Ollama if needed) ----
    import anthropic
    client = anthropic.Anthropic()                 # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=max_tokens, system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"{format_context(hits)}\n\nQuestion: {question}"}],  # context first, question last
    )
    # The response is a list of content blocks; join the text ones.
    text = "".join(b.text for b in msg.content if b.type == "text")
    # ---------------------------------------------------------------------------

    # Citation audit: pull every [TICKER-YYYYQn-tN-N] id out of the answer and
    # compare it with what was actually retrieved.
    cited = set(re.findall(r"\[([A-Z0-9.\-]+-\d{4}Q\d-t\d+-\d+)\]", text))
    provided = {c.id for c, _ in hits}
    return {
        "answer": text,
        "sources": [asdict(c) | {"score": s} for c, s in hits],   # dataclass -> dict, plus score
        "uncited_ids": sorted(provided - cited),       # retrieval noise signal
        "hallucinated_ids": sorted(cited - provided),  # grounding failure signal
    }


# ================================================================ 5. BUILD HELPERS
def build_index(pairs: list[tuple[str, str]], **kw) -> HybridIndex:
    """
    In-memory option: fetch + chunk several transcripts and index them in this process.

    pairs: [("IBM", "2026Q2"), ("IBM", "2026Q1"), ("MSFT", "2026Q2"), ...]
    **kw:  passed to HybridIndex, e.g. use_dense=False for a fast BM25-only index.
    """
    chunks = []
    for sym, q in pairs:
        chunks += chunk_transcript(fetch_transcript(sym, q), sym, q)
    return HybridIndex(chunks, **kw)


def build_retriever(db: str = VECTOR_DB, pairs: list[tuple[str, str]] | None = None,
                    embedder=None, use_rerank=True, recreate=False):
    """
    One entry point for every store. Returns an object with .search(query, k, filters).

      db="memory"                  -> builds a HybridIndex from `pairs` (required)
      db="milvus"/"qdrant"/"pinecone" -> opens the database; if `pairs` is given,
                                      ingests them first (safe to repeat; upserts are idempotent)
    """
    if db == "memory":
        if not pairs:
            raise ValueError("--db memory has no saved data: pass the transcripts with --pairs")
        return build_index(pairs, use_rerank=use_rerank)
    embedder = embedder or Embedder()
    store = make_store(db)
    if pairs:
        ingest(store, embedder, pairs, recreate=recreate)
    return StoreRetriever(store, embedder, use_rerank=use_rerank)


# ================================================================ 6. COMMAND LINE
def parse_pairs(items) -> list[tuple[str, str]]:
    """["IBM:2026Q2", "msft:2026Q1"] -> [("IBM", "2026Q2"), ("MSFT", "2026Q1")]"""
    out = []
    for p in items or []:
        sym, _, q = p.partition(":")
        if not re.fullmatch(r"\d{4}Q[1-4]", q):
            raise SystemExit(f"Bad pair {p!r}; expected TICKER:YYYYQn, e.g. IBM:2026Q2")
        out.append((sym.upper(), q))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Earnings-call RAG with a selectable vector store")
    ap.add_argument("--db", choices=DB_CHOICES, default=VECTOR_DB,
                    help=f"vector store to use (default: {VECTOR_DB}; set VECTOR_DB to change)")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("ingest", help="fetch transcripts and store them in the database")
    p_in.add_argument("pairs", nargs="+", help="TICKER:YYYYQn, e.g. IBM:2026Q2")
    p_in.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")

    p_ask = sub.add_parser("ask", help="retrieve excerpts and (optionally) generate an answer")
    p_ask.add_argument("question")
    p_ask.add_argument("--symbol"); p_ask.add_argument("--quarter"); p_ask.add_argument("--section",
                       choices=["prepared", "qa"])
    p_ask.add_argument("-k", type=int, default=8, help="excerpts sent to the LLM")
    p_ask.add_argument("--pairs", nargs="*", help="transcripts to (re)ingest first; required for --db memory")
    p_ask.add_argument("--no-llm", action="store_true", help="print retrieved excerpts only")
    a = ap.parse_args(argv)

    if a.cmd == "ingest":
        if a.db == "memory":
            raise SystemExit("--db memory does not persist anything; use 'ask --pairs ...' instead")
        store, emb = make_store(a.db), Embedder()
        n = ingest(store, emb, parse_pairs(a.pairs), recreate=a.recreate)
        print(f"[{a.db}] upserted {n} chunks; collection now holds {store.count()}")
        return

    retriever = build_retriever(a.db, parse_pairs(a.pairs), use_rerank=not a.no_rerank)
    filters = {f: getattr(a, f) for f in ("symbol", "quarter", "section") if getattr(a, f)}
    if a.symbol:
        filters["symbol"] = a.symbol.upper()

    if a.no_llm:
        for c, s in retriever.search(a.question, k=a.k, filters=filters):
            print(f"{s:8.4f}  {c.id:<22} [{c.section}] {c.speaker}: {c.text[:110]}...")
        return

    out = answer(retriever, a.question, k=a.k, filters=filters)
    print(out["answer"])
    print("\nSources:", [s["id"] for s in out["sources"]])
    if out.get("hallucinated_ids"):
        print("WARNING: cited ids that were not retrieved:", out["hallucinated_ids"])


if __name__ == "__main__":
    main()