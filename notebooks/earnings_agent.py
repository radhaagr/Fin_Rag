"""
Conversational agent over stored earnings reports: Postgres holds structured
report data (including each report's table as JSONB), ChromaDB holds
embeddings for retrieval, and a LangGraph ReAct agent answers user queries
by retrieving the right report then running the composable operation-
language pipeline (table_program_pipeline.py) against its table -- falling
back to plain text reasoning when a report has no usable table.

This is the storage + agent layer PortfolioAgent already runs on (Postgres +
ChromaDB + LangGraph) -- this module adds an earnings-report-specific tool
and ingestion path on top of that same stack, not a new stack.

Requires: pip install psycopg2-binary chromadb sentence-transformers \
    langgraph langchain-anthropic --break-system-packages
Postgres: a running instance, connection string via PG_DSN env var.
"""

import json
import os
import re

import chromadb
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from table_comparison_pipeline import parse_finqa_table
from table_program_pipeline import answer_query, make_client, _call_llm

PG_DSN = os.environ.get("PG_DSN", "dbname=portfolio_agent password=1234 user=postgres")
CHROMA_DIR = "./earnings_chroma"
CHROMA_COLLECTION = "earnings_reports"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"
LLM_PROVIDER = "anthropic"


# ---------------------------------------------------------------------------
# Storage layer: Postgres (structured) + Chroma (retrieval)
# ---------------------------------------------------------------------------

def _parse_dsn_params(dsn: str) -> dict:
    """Parses a simple space-separated 'key=value key2=value2' DSN string
    (the format used throughout this module) into a dict -- needed to swap
    out just the dbname while keeping host/user/password/port intact."""
    params = {}
    for part in dsn.split():
        if "=" in part:
            key, _, value = part.partition("=")
            params[key] = value
    return params


def _build_dsn(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def ensure_database_exists(dsn: str) -> None:
    """Creates the database named in `dsn` if it doesn't already exist.
    Connects to Postgres's default 'postgres' database to do this -- you
    can't run CREATE DATABASE while connected to the database you're
    creating, so this always uses a separate maintenance connection with
    the same host/user/password/port but dbname swapped to 'postgres'."""
    params = _parse_dsn_params(dsn)
    target_db = params.get("dbname")
    if not target_db:
        raise ValueError(f"No dbname found in dsn: {dsn!r}")

    maintenance_params = {**params, "dbname": "postgres"}
    maintenance_dsn = _build_dsn(maintenance_params)

    conn = psycopg2.connect(maintenance_dsn)
    conn.autocommit = True   # CREATE DATABASE can't run inside a transaction block
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if cur.fetchone() is not None:
                print(f"Database '{target_db}' already exists -- skipping.")
            else:
                cur.execute(f'CREATE DATABASE "{target_db}"')
                print(f"Created database '{target_db}'.")
    finally:
        conn.close()


def init_postgres_schema(dsn: str = PG_DSN) -> None:
    """Run once. Table field stored as JSONB -- the raw list-of-lists,
    same shape parse_finqa_table already expects, so no format conversion
    needed between storage and the reasoning pipeline.

    Creates the target database first if it doesn't exist yet -- no need
    to call ensure_database_exists separately before this."""
    ensure_database_exists(dsn)
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS earnings_reports (
                id TEXT PRIMARY KEY,
                company TEXT,
                fiscal_period TEXT,
                pre_text TEXT,
                post_text TEXT,
                table_data JSONB,          -- NULL if this report has no structured table
                raw_text TEXT NOT NULL,     -- always present; text-fallback reasoning uses this
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        conn.commit()


def serialize_table_for_embedding(table: list[list[str]]) -> str:
    """Same label-rich serialization from finqa_retrieval_and_reasoning.py --
    row labels need to be IN the embedded text for cosine similarity to have
    anything meaningful to match a query against."""
    header = table[0][1:]
    lines = []
    for row in table[1:]:
        cells = "; ".join(f"{col}: {val}" for col, val in zip(header, row[1:]))
        lines.append(f"{row[0]} — {cells}")
    return "\n".join(lines)


def ingest_report(report: dict, dsn: str = PG_DSN,
                   chroma_dir: str = CHROMA_DIR, embed_model: str = EMBED_MODEL) -> None:
    """report: {"id": ..., "company": ..., "fiscal_period": ...,
    "pre_text": [...], "post_text": [...], "table": [[...]] or None}.
    Writes structured data to Postgres AND an embedding to Chroma in one call
    -- this is the single ingestion path; there is no route to add a report
    to one store without the other, so they can't silently drift apart."""
    report_id = report.get("id", "<missing id>")
    print(f"[ingest_report] START id={report_id}")

    pre_text = " ".join(report.get("pre_text", []))
    post_text = " ".join(report.get("post_text", []))
    table = report.get("table")
    table_text = serialize_table_for_embedding(table) if table else ""
    raw_text = f"{pre_text}\n{table_text}\n{post_text}".strip()
    print(f"[ingest_report] id={report_id} built raw_text ({len(raw_text)} chars, "
          f"has_table={table is not None})")

    print(f"[ingest_report] id={report_id} connecting to Postgres ...")
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        print(f"[ingest_report] id={report_id} Postgres connected, executing INSERT ...")
        cur.execute("""
            INSERT INTO earnings_reports (id, company, fiscal_period, pre_text, post_text, table_data, raw_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                company = EXCLUDED.company, fiscal_period = EXCLUDED.fiscal_period,
                pre_text = EXCLUDED.pre_text, post_text = EXCLUDED.post_text,
                table_data = EXCLUDED.table_data, raw_text = EXCLUDED.raw_text
        """, (report["id"], report.get("company"), report.get("fiscal_period"),
              pre_text, post_text, json.dumps(table) if table else None, raw_text))
        conn.commit()
    print(f"[ingest_report] id={report_id} Postgres INSERT committed")

    # Reuses the module-level cached embedder (see _get_embedder below) --
    # previously this constructed a brand-new SentenceTransformer on EVERY
    # call, reloading the model from disk for every single report in a
    # batch. Loaded once now, on the first call only.
    print(f"[ingest_report] id={report_id} getting embedder (cached after first call) ...")
    embedder = _get_embedder(embed_model)
    print(f"[ingest_report] id={report_id} encoding text ...")
    vec = embedder.encode([raw_text], normalize_embeddings=True)[0].tolist()
    print(f"[ingest_report] id={report_id} embedding done (dim={len(vec)})")

    print(f"[ingest_report] id={report_id} connecting to Chroma at {chroma_dir} ...")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    print(f"[ingest_report] id={report_id} upserting into Chroma collection "
          f"'{CHROMA_COLLECTION}' ...")
    collection.upsert(
        ids=[report["id"]], embeddings=[vec], documents=[raw_text],
        metadatas=[{"company": report.get("company", ""), "fiscal_period": report.get("fiscal_period", "")}],
    )
    print(f"[ingest_report] id={report_id} DONE (Postgres + Chroma both written)")


def get_report(report_id: str, dsn: str = PG_DSN) -> dict | None:
    with psycopg2.connect(dsn) as conn:
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM earnings_reports WHERE id = %s", (report_id,))
            row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Retrieval + reasoning tool
# ---------------------------------------------------------------------------

_embedders: dict = {}   # cache keyed by model name -- avoids silently ignoring
                         # a non-default embed_model passed to ingest_report
_chroma_collection = None


def _get_embedder(model_name: str = EMBED_MODEL):
    if model_name not in _embedders:
        _embedders[model_name] = SentenceTransformer(model_name)
    return _embedders[model_name]


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return _chroma_collection


def answer_from_text(client, question: str, raw_text: str) -> dict:
    prompt = (f"Using ONLY this text, answer the question with a single number "
              f"or a short direct sentence if it's not a numeric question.\n\n"
              f"{raw_text}\n\nQuestion: {question}\n\nFinal Answer:")
    raw = _call_llm(client, LLM_PROVIDER, LLM_MODEL, prompt)
    return {"predicted": raw, "answer_source": "text_fallback"}


@tool
def answer_earnings_query(question: str, top_k: int = 3) -> str:
    """Answer a question about stored earnings reports. Retrieves the most
    relevant report(s), and uses table-aware numerical reasoning when the
    report has structured table data, falling back to text-based reasoning
    otherwise. Use this for any question about specific financial figures,
    comparisons, trends, or facts from earnings reports."""
    embedder = _get_embedder()
    collection = _get_chroma_collection()
    query_vec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    candidate_ids = results["ids"][0]

    if not candidate_ids:
        return "No relevant earnings reports found."

    client = make_client(LLM_PROVIDER)

    for doc_id in candidate_ids:
        report = get_report(doc_id)
        if report is None:
            continue

        if report.get("table_data"):
            df = parse_finqa_table(report["table_data"])
            outcome = answer_query(client, LLM_PROVIDER, LLM_MODEL, df, question)
            if outcome["predicted"] is not None:
                return (f"Answer: {outcome['predicted']}\n"
                        f"Source: {report.get('company', doc_id)} ({report.get('fiscal_period', 'unknown period')})\n"
                        f"Reasoning trace: {outcome['trace']}")
        else:
            outcome = answer_from_text(client, question, report["raw_text"])
            return (f"Answer: {outcome['predicted']}\n"
                    f"Source: {report.get('company', doc_id)} ({report.get('fiscal_period', 'unknown period')})\n"
                    f"(answered from unstructured text -- no table available for this report)")

    return ("Found potentially relevant reports but could not compute a confident answer. "
            "Consider rephrasing the question or checking whether the needed figure exists in the data.")


# ---------------------------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial research assistant with access to stored earnings reports. "
    "Always call answer_earnings_query for questions about specific figures, comparisons, "
    "or trends -- do not answer from memory. Relay the tool's answer clearly, and mention "
    "the source report. If the tool cannot find a confident answer, say so explicitly."
)

llm = ChatAnthropic(model=LLM_MODEL)
checkpointer = InMemorySaver()
earnings_agent = create_react_agent(model=llm, tools=[answer_earnings_query],
                                     prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


def chat(message: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = earnings_agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return result["messages"][-1].content


if __name__ == "__main__":
    init_postgres_schema()

    # Example ingestion -- replace with your actual FinQA/Kaggle-sourced reports
    example_report = {
        "id": "AAPL_2017_10K",
        "company": "Apple",
        "fiscal_period": "FY2017",
        "pre_text": ["the following table summarizes selected financial data ."],
        "post_text": ["all figures in millions of dollars ."],
        "table": [
            ["", "2016", "2017"],
            ["net income", "45687", "48351"],
            ["total revenue", "215639", "229234"],
        ],
    }
    ingest_report(example_report)
    print("Ingested example report.")

    print("\nEarnings report assistant. Type 'quit' to exit.")
    thread_id = "cli-session"
    while True:
        msg = input("You: ")
        if msg.strip().lower() == "quit":
            break
        print("Agent:", chat(msg, thread_id=thread_id))