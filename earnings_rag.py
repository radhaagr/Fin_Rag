"""
Earnings-call transcript RAG (Retrieval-Augmented Generation)
=============================================================

Pipeline:
  1. FETCH     - download a transcript from Alpha Vantage (cached to disk)
  2. CHUNK     - split it into speaker-aware pieces with metadata
  3. INDEX     - build a hybrid index: BM25 (keywords) + dense embeddings (meaning)
  4. RETRIEVE  - fuse both rankings with RRF, optionally rerank with a cross-encoder
  5. GENERATE  - ask an LLM to answer ONLY from retrieved excerpts, with citations

Install:
  pip install requests rank_bm25 numpy sentence-transformers anthropic

Environment variables:
  ALPHAVANTAGE_API_KEY  - required for fetching transcripts
  ANTHROPIC_API_KEY     - required for answer generation
  LLM_MODEL             - optional, overrides the default model
  EMBED_MODEL           - optional, overrides the embedding model
  RERANK_MODEL          - optional, overrides the reranker
  TRANSCRIPT_CACHE      - optional, folder for cached transcripts
"""
from __future__ import annotations   # allows "list[dict]" style type hints on older Python
import json, os, re, pathlib
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


# ================================================================ 3. INDEX
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

    Fine for tens to a few hundred transcripts. Beyond that, move vectors to
    pgvector / Qdrant / Elasticsearch.
    """
    def __init__(self, chunks: list[Chunk], use_dense=True, use_rerank=True):
        self.chunks = chunks
        # Prefix each passage with company/quarter/speaker/role so those words are
        # searchable too (e.g. a query mentioning "CFO" or "2026Q2" can match lexically).
        self.docs = [f"{c.symbol} {c.quarter} {c.speaker} ({c.title}): {c.text}" for c in chunks]

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


def answer(index: HybridIndex, question: str, k=8, filters=None, max_tokens=1200) -> dict:
    """
    Retrieve relevant excerpts, ask the LLM, and audit its citations.

    Returns a dict:
        answer           - the LLM's text answer
        sources          - the retrieved chunks (with scores) that were sent to the LLM
        uncited_ids      - retrieved but unused chunks (high count = noisy retrieval)
        hallucinated_ids - ids the LLM cited that were NOT provided (should always be empty)

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


# ================================================================ 5. BUILD HELPER
def build_index(pairs: list[tuple[str, str]], **kw) -> HybridIndex:
    """
    Fetch + chunk several transcripts and index them together.

    pairs: [("IBM", "2026Q2"), ("IBM", "2026Q1"), ("MSFT", "2026Q2"), ...]
    **kw:  passed to HybridIndex, e.g. use_dense=False for a fast BM25-only index.
    """
    chunks = []
    for sym, q in pairs:
        chunks += chunk_transcript(fetch_transcript(sym, q), sym, q)
    return HybridIndex(chunks, **kw)


# ================================================================ Example run
#if __name__ == "__main__":
    # Index two IBM quarters, then ask a question restricted to Q2 2026.
idx = build_index([("IBM", "2026Q2"), ("IBM", "2026Q1")])
out = answer(idx, "What caused the software shortfall, and is the demand deferred or lost?",
             filters={"symbol": "IBM", "quarter": "2026Q2"})
print(out["answer"])
print("\nSources:", [s["id"] for s in out["sources"]])
