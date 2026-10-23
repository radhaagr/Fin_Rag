"""
FinanceRAG (ACM ICAIF-24 challenge) adapter for the FinanceBench harness
=======================================================================
Runs on the competition's BEIR-style files for the FinanceBench task:
    corpus.jsonl   {"_id": "...", "title": "...", "text": "..."}   one line per chunk/page
    queries.jsonl  {"_id": "...", "text": "..."}                     the FinanceBench questions
    qrels (opt.)   TSV/CSV  query_id, corpus_id, score               relevance labels, if you have them

What it adds on top of financebench_starter.py
  match_queries   joins each query to its FinanceBench row (answer, type, gold pages) by question text
  derive_qrels    builds relevance labels from FinanceBench's gold evidence pages by text matching,
                  for any chunk size; cross-checks official qrels if you have them
  evaluate        nDCG / Recall / P / MAP @k and MRR@10, same definitions as the official pytrec_eval
                  code, averaged two ways: official (queries with results) and strict (missing = 0)
  bm25_run        BM25 baseline (needs `pip install bm25s`); filing names and FY periods are split into words
  write_submission  Kaggle format: query_id,corpus_id, top-10 rows per query in rank order
  answers_to_preds  turns FinanceRAG answers {"query_id","answer"} into financebench_starter predictions
  retrieve        rankers: bm25 (the baseline), dense (an embedding model), hybrid (both, fused by rank)
  route_scores    filing routing: passages of the filing a question names (company, year, quarter) go first
  compare         several rankers scored against the same answer key, in one table with 95% intervals

CLI (the data root is searched for the task's files, whatever the layout)
  python finrag_eval.py find     --data-root D:\\Fin_Rag\\finrag_data
  python finrag_eval.py baseline --data-root D:\\Fin_Rag\\finrag_data --fb financebench --out out
  python finrag_eval.py baseline --data-root D:\\Fin_Rag\\finrag_data --fb financebench --method hybrid --route
  python finrag_eval.py compare  --data-root D:\\Fin_Rag\\finrag_data --fb financebench
  python finrag_eval.py eval     --data-root D:\\Fin_Rag\\finrag_data --fb financebench --run results.csv
      --task-dir DIR  instead of --data-root, if corpus.jsonl and queries.jsonl sit directly in DIR
      --qrels FILE    official relevance labels; picked up automatically under --data-root if present
      --run FILE      Kaggle/FinanceRAG results.csv, a TREC run, or JSONL {"query_id","corpus_ids"}
      --fb DIR        folder made by `financebench_starter.py setup` (derived qrels, per-type results)
      --trace         print every step of the code flow and the data passed between steps
      --method M      baseline: bm25 (default), dense or hybrid; dense and hybrid need `pip install sentence-transformers`
      --route         baseline: add filing routing to the method
      --model NAME    embedding model for dense and hybrid (default BAAI/bge-small-en-v1.5)
      --methods LIST  compare: which rankers, e.g. bm25,bm25+route,hybrid+route (default: all six); --out adds compare.csv

Notebook (raw string for Windows paths: in "D:\\Fin_Rag\\finrag_data" the \\f is a form feed)
  import finrag_eval as fr
  res = fr.main()           # no arguments: runs NOTEBOOK_ARGS (set below) and traces every step
  res = fr.main(["baseline", "--data-root", r"D:\\Fin_Rag\\finrag_data", "--fb", "financebench", "--out", "out"])
  Code pasted into a cell instead of imported: running that cell only defines the functions;
  call res = main() in the next cell (financebench_starter.py in the notebook's folder, or pasted above).

HOW THE CODE FLOWS  (read this first; the STEP numbers match the section banners below)
  main(argv)                                          STEP 6  read the command-line arguments, then:
   1. find the files    find_task_files()             STEP 1  (or --task-dir / --corpus-file + --queries-file)
   2. load them         load_corpus(), load_queries() STEP 1
   3. get a ranking     baseline: retrieve()          STEP 4b bm25_run() for plain BM25; otherwise a score matrix from
                                                              _bm25_matrix() / _dense_matrix() (+ _rrf() for hybrid),
                                                              then route_scores() with --route; write_submission() if --out
                        eval:     load_run()          STEP 1  (your retriever's results file)
                        compare:  compare()           STEP 5  retrieve() + evaluate() for each method, one table, then stop
   4. score it          run_eval()                    STEP 5
        a. with --fb    match_queries()     which FinanceBench question each query is   STEP 3
                        derive_qrels()      answer key for every query, from the gold evidence
                        evaluate(), print_summary(), by_question_type()                STEP 2
        b. qrels file   load_qrels() -> evaluate() -> print_summary()                   STEP 1, 2
                        compare_qrels()     official answer key vs the derived one
        c. with --out   write_per_query(), qrels_derived.tsv                            STEP 4, 5
  To watch it happen, turn on TRACE (fr.main() with no arguments in a notebook does): each step prints
  "-> function(inputs)" when it starts and "<- function: output [seconds]" when it ends, and one query
  (TRACE_QUERY) is followed from its question text to its score.

THE DATA SHAPES PASSED AROUND  (ids from your FinanceBench task files; scores made up)
  corpus   {"dd2abb8b8": {"title": "3M_2018_10K", "text": "..."}}       passage id -> passage
  queries  {"qd2abb7e6": "...what is the year end FY2018 net PPNE for 3M?..."}  query id -> question
  run      {"qd2abb7e6": {"dd2abb8b8": 14.2, "dd2ac5228": 9.8, ...}}    your system's scores; higher = ranked earlier
  qrels    {"qd2abb7e6": {"dd2abb8b8": 1}}                              answer key; 1 = relevant
  qmap     {"qd2abb7e6": "financebench_id_04672"}                       competition query id -> FinanceBench id
"""
from __future__ import annotations
import argparse, csv, difflib, functools, json, math, os, re, sys, time
from collections import Counter, defaultdict

# Cut-offs at which every metric is computed (nDCG@1, nDCG@3, ... nDCG@100). The competition ranks on nDCG@10.
K_VALUES = (1, 3, 5, 10, 20, 100)

# What main() runs when a notebook cell calls it with no arguments: res = fr.main(). Edit to match your folders,
# or set fr.NOTEBOOK_ARGS = [...] before calling. "financebench" and "out" are relative to the notebook's folder.
# Add "--route" for filing routing, "--method", "hybrid" for BM25 + embeddings, or start with "compare" instead of
# "baseline" to score every method side by side.
NOTEBOOK_ARGS = ["baseline", "--data-root", r"D:\Fin_Rag\finrag_data", "--fb", "financebench", "--out", "out"]


def _in_notebook() -> bool:
    # True in Jupyter/IPython or at an interactive >>> prompt, where sys.argv is not our command line.
    return "ipykernel" in sys.modules or hasattr(sys, "ps1")


# =========================================================================== TRACE: watch the code flow (learning aid)
# With TRACE on, each main step prints "-> function(inputs)" when it starts and "<- function: output [seconds]" when
# it ends, indented by call depth, plus a few lines showing the data at that point. One query, TRACE_QUERY, is
# followed through every step. main() turns TRACE on for a no-argument notebook run, with --trace, or trace=True.
TRACE = False
TRACE_QUERY = "qd2abb7e6"   # the 3M FY2018 net PP&E question in your files; if absent, the first query is followed
_depth = 0                  # current call depth, for indenting trace lines
_tq = None                  # the query followed in this run (main sets it once the queries are loaded)


def _say(msg: str) -> None:
    # One trace line, indented to the current call depth. Prints nothing unless TRACE is on.
    if TRACE:
        print("  " * _depth + msg)


def _short(x, n: int = 70, deep: bool = True) -> str:
    # One-line preview of any value, so a trace line never floods the screen. deep=False shows only a container's
    # type and size (used for a function's inputs); deep=True also shows its first item (used for outputs).
    if isinstance(x, str):
        s = " ".join(x.split())
        if len(s) > n and " " not in s and ("/" in s or "\\" in s):  # a long path: its end says the most
            return "'..." + s[-n:] + "'"
        return "'" + (s if len(s) <= n else s[:n] + "...") + "'"
    if isinstance(x, (dict, list, tuple)) and not deep:
        return f"{type(x).__name__}[{len(x):,}]"
    if isinstance(x, dict):
        if not x:
            return "{}"
        k, v = next(iter(x.items()))
        return f"dict[{len(x):,}] {{{_short(k, 30)}: {_short(v, n // 2)}, ...}}"
    if isinstance(x, tuple) and len(x) <= 3:            # a short tuple is usually several return values
        return "(" + ", ".join(_short(v, n // 2) for v in x) + ")"
    if isinstance(x, (list, tuple)):
        return f"list[{len(x):,}] [{_short(x[0], n // 2)}, ...]" if x else "[]"
    if isinstance(x, (set, frozenset)):
        return f"set[{len(x):,}]"
    if hasattr(x, "shape") and hasattr(x, "dtype"):     # a numpy score matrix: its shape, not its numbers
        return f"array[{' x '.join(str(d) for d in x.shape)}] {x.dtype}"
    if isinstance(x, float):
        return f"{x:.4f}"
    return repr(x)


def _traced(fn):
    # Decorator for the main steps: with TRACE on, print the call and its inputs, then its output and run time.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _depth
        if not TRACE:
            return fn(*args, **kwargs)
        names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        shown = [f"{k}={_short(v, 40, deep=False)}" for k, v in list(zip(names, args)) + list(kwargs.items())]
        _say(f"-> {fn.__name__}({', '.join(shown)})")
        _depth += 1
        start = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        finally:
            _depth -= 1
        _say(f"<- {fn.__name__}: {_short(out)}   [{time.perf_counter() - start:.2f}s]")
        return out
    return wrapper


# =========================================================================== STEP 1: find and load files
def _key(s: str) -> str:
    # Lowercase and keep only a-z0-9, so "FinanceBench/corpus.jsonl", "financebench_corpus.jsonl" and
    # "FINANCEBENCH CORPUS" all compare equal: "financebenchcorpusjsonl".
    return re.sub(r"[^a-z0-9]", "", s.lower())


@_traced
def find_task_files(root: str, task: str = "FinanceBench") -> dict[str, str | None]:
    """Locate a task's corpus / queries / qrels under a data root, whatever the folder layout, e.g.
    root/FinanceBench/corpus.jsonl, root/FinanceBench_corpus.jsonl, root/qrels/FinanceBench_qrels.tsv.
    `root` may also be the task folder itself. Windows paths work: use r"D:\\Fin_Rag\\finrag_data".

    Returns {"corpus": path, "queries": path, "qrels": path or None}.
    For every file under root it decides two things:
      - which task it belongs to: the longest task name that appears in its path (the "owner")
      - what kind of file it is: .jsonl with "corpus" or "queries" in the path, .tsv/.csv with "qrel"
    Files of the requested task go into `found`. Files that name no task at all (a flat folder holding
    corpus.jsonl, like yours) go into `generic`, which is only used when nothing task-named exists."""
    root = os.path.abspath(os.path.expanduser(root))          # "~" and relative paths -> full path
    if not os.path.isdir(root):
        raise FileNotFoundError(f"data root not found: {root!r} (on Windows use a raw string r'D:\\...' or forward slashes)")
    t = _key(task)                                            # "FinanceBench" -> "financebench"
    tasks = {_key(o) for o in ("FinDER", "FinQABench", "FinQA", "TATQA", "ConvFinQA", "MultiHiertt", "FinanceBench")} | {t}
    found = {"corpus": [], "queries": [], "qrels": []}
    generic = {"corpus": [], "queries": []}          # e.g. root/corpus.jsonl when only one task was downloaded
    n_files = 0                                               # trace only: files looked at so far
    for dirpath, _, files in os.walk(root):                   # every file in every sub-folder
        for f in files:
            full = os.path.join(dirpath, f)
            rel = _key(os.path.relpath(full, root))           # path below root, normalised by _key()
            named = [k for k in tasks if k in rel]            # task names that occur in that path
            owner = max(named, key=len) if named else None   # longest name wins: FinQABench is not FinQA
            # The file is ours if our task owns it, or if it names no task and root itself is the task folder.
            in_task = owner == t or (owner is None and t in _key(os.path.basename(root)))
            if f.endswith((".part", ".zip")) or (owner is not None and owner != t):
                decision = "skipped (another task, an archive or a partial download)"
            else:
                kind = ("corpus" if f.endswith(".jsonl") and "corpus" in rel else
                        "queries" if f.endswith(".jsonl") and ("queries" in rel or "query" in rel) else
                        "qrels" if f.endswith((".tsv", ".csv")) and any(s in rel for s in ("qrel", "groundtruth", "gt", "label"))
                        else None)
                if kind and in_task:
                    found[kind].append(full)
                    decision = f"{kind} file of {task}"
                elif kind in generic:                         # a corpus/queries file that names no task
                    generic[kind].append(full)
                    decision = f"{kind} file naming no task (used only if no file names {task})"
                else:
                    decision = "ignored (not a corpus, queries or qrels file of this task)"
            n_files += 1
            if n_files <= 15:                                 # trace the first 15 files only
                _say(f"{os.path.relpath(full, root)}  [task named in path: {owner or 'none'}]  -> {decision}")
    if n_files > 15:
        _say(f"... and {n_files - 15} more files")
    out = {}
    for kind in ("corpus", "queries"):
        # Prefer files named after the task; among several, the shortest path (closest to root) wins.
        c = sorted(found[kind], key=len) or sorted(generic[kind], key=len)
        if not c:
            raise FileNotFoundError(f"no {task} {kind} .jsonl found under {root}")
        if not found[kind]:
            print(f"note: no file named after {task}; using {c[0]}")
        if len(c) > 1:
            print(f"note: several {kind} files; using the first of {c}")
        out[kind] = c[0]
    # qrels are optional: only a file that names the task counts, so another task's labels are never picked up.
    out["qrels"] = sorted(found["qrels"], key=len)[0] if found["qrels"] else None
    if len(found["qrels"]) > 1:
        print(f"note: several qrels files for {task}; using {out['qrels']} of {found['qrels']}")
    return out


def _read_jsonl(path: str) -> list[dict]:
    # JSON Lines: one JSON object per line; blank lines are skipped.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@_traced
def load_corpus(path: str) -> dict[str, dict]:
    """corpus.jsonl -> {passage_id: {"title": ..., "text": ...}}. Ids are kept as strings."""
    corpus = {}
    for r in _read_jsonl(path):
        cid = str(r["_id"])
        if cid in corpus:                                     # a repeated id would silently overwrite a passage
            raise ValueError(f"duplicate corpus _id {cid!r} in {path}")
        corpus[cid] = {"title": r.get("title") or "", "text": r.get("text") or ""}
    return corpus


@_traced
def load_queries(path: str) -> dict[str, str]:
    """queries.jsonl -> {query_id: question text}."""
    return {str(r["_id"]): r.get("text") or "" for r in _read_jsonl(path)}


def _table_rows(path: str) -> list[list[str]]:
    # Read a TSV, CSV or whitespace-separated file into rows of strings. The separator is guessed from
    # the first 4 KB: a tab means TSV, else a comma means CSV, else split on spaces (TREC files).
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(4096); f.seek(0)
        delim = "\t" if "\t" in sample else ("," if "," in sample else None)
        if delim:
            return [row for row in csv.reader(f, delimiter=delim) if row]
        return [line.split() for line in f if line.strip()]


@_traced
def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """TSV/CSV with header (query_id|query-id, corpus_id|corpus-id, score) or headerless
    3-column (qid, docid, score) / 4-column TREC (qid, 0, docid, rel).
    Returns the answer key {query_id: {corpus_id: relevance}}, e.g. {"qd2abb7e6": {"dd2abb8b8": 1}}."""
    rows = _table_rows(path)
    head = [h.strip().lower() for h in rows[0]]
    # If the last cell of the first row is not a number, that row is a header ("score"), not data.
    has_header = not re.fullmatch(r"-?\d+(\.\d+)?", head[-1])
    if has_header:
        # Find each column by its name, so column order doesn't matter.
        qi = next(i for i, h in enumerate(head) if "query" in h or h == "qid")
        di = next(i for i, h in enumerate(head) if "corpus" in h or "doc" in h or h == "pid")
        si = next((i for i, h in enumerate(head) if "score" in h or "rel" in h or "label" in h), None)
        rows = rows[1:]
    else:
        # No header: TREC qrels have 4 columns (qid, iteration, docid, relevance); else (qid, docid[, score]).
        qi, di, si = (0, 2, 3) if len(head) == 4 else (0, 1, 2 if len(head) > 2 else None)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        qrels[r[qi].strip()][r[di].strip()] = int(float(r[si])) if si is not None else 1
    return dict(qrels)


@_traced
def load_run(path: str) -> dict[str, dict[str, float]]:
    """Kaggle/FinanceRAG results.csv (query_id,corpus_id in rank order), TREC run
    (qid Q0 docid rank score tag), or JSONL {"query_id": ..., "corpus_ids": [...]} in rank order.
    All three become {query_id: {corpus_id: score}}. Formats that give only an order get score = -rank
    (0, -1, -2, ...), so "higher score = ranked earlier" holds everywhere downstream."""
    run: dict[str, dict[str, float]] = defaultdict(dict)
    if path.endswith(".jsonl"):                                          # JSONL: ids already in rank order
        _say("format: JSONL (query_id + corpus_ids in rank order) -> score = -rank")
        for r in _read_jsonl(path):
            ids = r.get("corpus_ids") or r.get("retrieved") or []
            run[str(r["query_id"])] = {str(c): -float(i) for i, c in enumerate(ids)}
        return dict(run)
    rows = _table_rows(path)
    if len(rows[0]) >= 6 and rows[0][1] in ("Q0", "0", "q0"):          # TREC run: real scores in column 5
        _say("format: TREC run (qid Q0 docid rank score tag) -> scores from column 5")
        for r in rows:
            run[r[0]][r[2]] = float(r[4])
        return dict(run)
    head = [h.strip().lower() for h in rows[0]]
    if "query_id" in head or "query-id" in head:                         # drop the CSV header row
        rows = rows[1:]
    _say("format: CSV query_id,corpus_id in rank order -> score = -rank")
    rank = Counter()                                                     # next rank number, per query
    for r in rows:                                                       # CSV in rank order
        q, c = r[0].strip(), r[1].strip()
        if c not in run[q]:                                              # keep the first (best) position
            run[q][c] = -float(rank[q]); rank[q] += 1
    return dict(run)


# =========================================================================== STEP 2: metrics
def _ranked(run_q: dict[str, float]) -> list[str]:
    # {passage_id: score} -> list of passage ids, best first.
    # trec_eval order: score descending, ties broken by doc id descending
    # (same as the official pytrec_eval scorer, so tied scores give identical metrics).
    return [d for d, _ in sorted(run_q.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def per_query_metrics(rels: dict[str, int], ranked: list[str], k_values=K_VALUES) -> dict[str, float]:
    """All metrics for ONE query.
      rels    this query's answer key, e.g. {"dd2abb8b8": 1}
      ranked  your system's passage ids, best first
    For each cut-off k:
      nDCG@k    DCG / ideal DCG. DCG adds relevance / log2(rank + 1) for every relevant passage in the
                top k, so a hit at rank 1 counts 1.0, rank 2 counts 0.63 and rank 10 counts 0.29. The ideal
                DCG is that sum if all relevant passages were ranked first, so 1.0 means a perfect order.
      Recall@k  share of the relevant passages that made the top k
      P@k       share of the top k that is relevant (divides by k even if fewer were returned, as trec_eval does)
      MAP@k     mean of the precision at each relevant hit within the top k, over all relevant passages
    MRR@10          1 / rank of the first relevant passage if it is in the top 10, else 0
    first_rel_rank  rank of the first relevant passage anywhere in the list (None if never retrieved)"""
    relevant = {d for d, g in rels.items() if g > 0}                     # ids marked relevant
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)    # the perfect ranking's gains
    out = {}
    for k in k_values:
        top = ranked[:k]
        # i is the 0-based position, so rank = i + 1 and the discount is log2(rank + 1) = log2(i + 2)
        dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top) if rels.get(d, 0) > 0)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        hits = [i for i, d in enumerate(top) if d in relevant]           # 0-based positions of the hits
        out[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
        out[f"Recall@{k}"] = len(hits) / len(relevant) if relevant else 0.0
        out[f"P@{k}"] = len(hits) / k
        # For the (j+1)-th hit at position i, precision at that rank is (j + 1) / (i + 1).
        out[f"MAP@{k}"] = (sum((j + 1) / (i + 1) for j, i in enumerate(hits)) / len(relevant)) if relevant else 0.0
    first = next((i for i, d in enumerate(ranked) if d in relevant), None)
    out["MRR@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
    out["first_rel_rank"] = (first + 1) if first is not None else None
    return out


@_traced
def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
             k_values=K_VALUES, ignore_identical_ids: bool = True):
    """Returns (summary, per_query). summary[metric] = {"official": mean over queries that have both
    qrels and results (what the FinanceRAG code reports), "strict": mean over all qrels queries,
    missing results counted as 0}. Matches pytrec_eval's ndcg_cut / recall / P / map_cut.
    per_query[query_id] = per_query_metrics(...), or None when the run has no results for that query."""
    per_query = {}
    for qid, rels in qrels.items():               # only queries that HAVE an answer key are scored
        if not any(g > 0 for g in rels.values()):  # no relevant passage at all: nothing to measure
            continue
        if qid not in run:                         # answer key exists but the run skipped this query
            per_query[qid] = None
            continue
        # BEIR convention: drop a passage whose id equals the query id (never true for FinanceRAG ids).
        ranked = [d for d in _ranked(run[qid]) if not (ignore_identical_ids and d == qid)]
        per_query[qid] = per_query_metrics(rels, ranked, k_values)
    have = [m for m in per_query.values() if m is not None]               # the queries actually scored
    metrics = [f"{m}@{k}" for m in ("nDCG", "Recall", "P", "MAP") for k in k_values] + ["MRR@10"]
    # "official" divides by the scored queries; "strict" divides by all of them, so skipped ones count as 0.
    summary = {m: {"official": sum(x[m] for x in have) / len(have) if have else 0.0,
                   "strict": sum(x[m] for x in have) / len(per_query) if per_query else 0.0} for m in metrics}
    summary["_coverage"] = {"qrels_queries": len(per_query), "with_results": len(have),
                            "run_queries_without_qrels": len(set(run) - set(qrels))}
    if TRACE and per_query.get(_tq) and "nDCG@10" in per_query[_tq]:   # how one query's score comes about
        top = [d for d in _ranked(run[_tq]) if not (ignore_identical_ids and d == _tq)][:10]
        m = per_query[_tq]
        _say(f"{_tq}: answer key {sorted(d for d, g in qrels[_tq].items() if g > 0)}")
        _say(f"{_tq}: top 10 (* = relevant) "
             + " ".join(f"{i}:{d}{'*' if qrels[_tq].get(d, 0) > 0 else ''}" for i, d in enumerate(top, 1)))
        _say(f"{_tq}: first relevant at rank {m['first_rel_rank'] or 'none (not retrieved)'} -> nDCG@10 {m['nDCG@10']:.2f}, "
             f"Recall@10 {m['Recall@10']:.2f}, MRR@10 {m['MRR@10']:.2f}")
    if "nDCG@10" in summary:
        _say(f"scored {len(have)} of {len(per_query)} labelled queries; mean nDCG@10 {summary['nDCG@10']['official']:.4f}")
    return summary, per_query


def print_summary(summary: dict, title: str = "", k_values=(1, 5, 10, 100)) -> None:
    # Prints the "official" averages as a small table; adds the strict nDCG@10 when queries were missing.
    cov = summary["_coverage"]
    print(f"\n=== {title}  (queries with qrels: {cov['qrels_queries']}, with results: {cov['with_results']})")
    print(f"   {'k':>4} {'nDCG':>7} {'Recall':>7} {'P':>7} {'MAP':>7}")
    for k in k_values:
        if f"nDCG@{k}" in summary:
            print(f"   {k:>4} " + " ".join(f"{summary[f'{m}@{k}']['official']:7.4f}" for m in ("nDCG", "Recall", "P", "MAP")))
    print(f"   MRR@10 {summary['MRR@10']['official']:.4f}")
    if cov["with_results"] < cov["qrels_queries"]:
        print(f"   strict nDCG@10 (missing queries = 0): {summary['nDCG@10']['strict']:.4f}")


# =========================================================================== STEP 3: join to FinanceBench, build the answer key
def _norm_q(s: str) -> str:
    # Curly quotes -> straight, runs of whitespace -> one space, lowercase: typography can't block a match.
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


@_traced
def match_queries(queries: dict[str, str], fb_rows: list[dict], cutoff: float = 0.9):
    """query_id -> financebench_id. Exact match on normalised text, then fuzzy (difflib >= cutoff).
    Pass 1 (exact) matches all 150 queries of your FinanceRAG files. Pass 2 only matters if a question
    was lightly edited: it takes the most similar unused FinanceBench question if similarity >= 0.9.
    Returns (qmap, info); info lists unmatched query ids and any FinanceBench id matched twice."""
    by_text = defaultdict(list)                       # normalised question -> [financebench_id, ...]
    for r in fb_rows:
        by_text[_norm_q(r["question"])].append(r["financebench_id"])
    qmap, unmatched, used = {}, [], Counter()
    for qid, text in queries.items():                 # pass 1: exact
        hits = by_text.get(_norm_q(text))
        if hits:
            qmap[qid] = hits[0]; used[hits[0]] += 1
    n_exact = len(qmap)                               # trace only: queries matched by pass 1
    # FinanceBench questions still free after pass 1: normalised text -> financebench_id
    left = {_norm_q(r["question"]): r["financebench_id"] for r in fb_rows if r["financebench_id"] not in used}
    for qid, text in queries.items():                 # pass 2: fuzzy, for whatever pass 1 missed
        if qid in qmap:
            continue
        best = difflib.get_close_matches(_norm_q(text), list(left), n=1, cutoff=cutoff)
        if best:
            qmap[qid] = left.pop(best[0])             # pop: each FinanceBench question is used once
        else:
            unmatched.append(qid)
    dup = [f for f, n in Counter(qmap.values()).items() if n > 1]
    _say(f"pass 1 (exact text): {n_exact} matched | pass 2 (fuzzy, similarity >= {cutoff}): {len(qmap) - n_exact}"
         f" | unmatched: {len(unmatched)}")
    if _tq in qmap:
        _say(f"{_tq} -> {qmap[_tq]}")
    return qmap, {"matched": len(qmap), "unmatched": unmatched, "fb_ids_matched_twice": dup}


_THOUSANDS = re.compile(r"(?<=\d)\s*,\s*(?=\d{3}\b)")      # the comma in "1,577" (or "1 , 577")
_MTOK = re.compile(r"\d+(?:\.\d+)?|[a-z]{2,}")             # numbers like 1577 or 5.1, and words of 2+ letters


def _mtoks(text: str) -> tuple[set, set]:
    """(words, numbers) used to find candidates. Joins thousands separators ('1,577', '1 , 577' -> '1577')."""
    toks = _MTOK.findall(_THOUSANDS.sub("", text.lower()))
    return {t for t in toks if t[0].isalpha()}, {t for t in toks if t[0].isdigit()}


def _shingles(text: str, k: int = 10) -> set:
    """Character 10-grams of the text with everything but a-z0-9 removed. Whitespace-insensitive on purpose:
    FinanceBench's evidence text drops non-breaking spaces ('Costofproductssold') while most parsers keep them.
    Example: "Cost of products sold" -> "costofproductssold" -> {"costofprod", "ostofprodu", ..., "roductssol"}.
    Two texts are compared by how many of these 10-grams they share."""
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    return {s[i:i + k] for i in range(len(s) - k + 1)}


@_traced
def derive_qrels(corpus: dict[str, dict], fb_rows: list[dict], qmap: dict[str, str],
                 member_min: float = 0.85, num_min: float = 0.8, min_shingles: int = 100,
                 n_candidates: int = 60):
    """Relevance labels from FinanceBench gold evidence pages, for any corpus chunking (page, 512 tokens, ...).
    A corpus entry is relevant to a query when
      (a) it lies on a gold evidence page: >= member_min of its character 10-grams occur on that page, and,
          if it has >= 3 numbers, >= num_min of them do too (this separates FY2018's statement from FY2019's,
          which share every label and two of three columns); entries under min_shingles are ignored, and
      (b) it overlaps the evidence: covers >= 50% of the evidence, or >= 80% of the entry lies inside it.
    Identical copies of a gold page elsewhere in the corpus also qualify; they are flagged in the diagnostics.
    First, though, an entry whose text IS the evidence text (whitespace-normalised), or failing that the full
    gold page, is taken as the label for that evidence item and the fuzzy rules are skipped for it. That is the
    case for the FinanceRAG FinanceBench corpus, which consists of FinanceBench's evidence_text snippets.
    Returns (qrels, diagnostics per gold evidence page).

    The work happens in three phases:
      A  one "item" per gold evidence entry (189 items for the 150 questions; some cite 2-3 pages)
      B  index the corpus once, so each item only compares itself with a few likely passages
      C  per item: exact match first; otherwise score up to 60 candidates against rules (a) and (b)"""
    fb = {r["financebench_id"]: r for r in fb_rows}
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()        # whitespace-insensitive text comparison

    # ---- PHASE A: what each gold evidence entry looks like, in the forms the matching needs
    items = []
    for qid, fid in qmap.items():
        for j, e in enumerate(fb[fid]["evidence"]):
            pw, pn = _mtoks(e["evidence_text_full_page"])        # words / numbers on the whole gold page
            items.append({"qid": qid, "j": j, "doc": e["doc_name"], "page": e["evidence_page_num"],
                          "toks": pw | pn,                       # used to find candidate passages (phase B/C)
                          "nums": pn,                            # rule (a), numbers part
                          "P": _shingles(e["evidence_text_full_page"]),   # rule (a): the gold page
                          "E": _shingles(e["evidence_text"]),             # rule (b): the evidence snippet
                          # texts that count as an exact match, in priority order: snippet, then full page
                          "exact_keys": [norm(e["evidence_text"]), norm(e["evidence_text_full_page"])]})
    _say(f"phase A: {len(items)} gold evidence items for {len(qmap)} matched queries")
    for it in (items if TRACE else []):
        if it["qid"] == _tq:
            _say(f"{_tq}: gold page {it['doc']} page {it['page']} (0-indexed) | "
                 f"10-grams: page {len(it['P']):,}, evidence {len(it['E']):,}")

    # ---- PHASE B: index the corpus
    ids = list(corpus)                                           # row number i <-> passage id ids[i]
    texts = [corpus[c]["title"] + "\n" + corpus[c]["text"] for c in ids]
    by_text = defaultdict(list)                                  # normalised passage text -> passage ids
    for c in ids:
        by_text[norm(corpus[c]["text"])].append(c)
    df = Counter()                                       # pass 1: document frequency of each token
    for t in texts:
        w, n = _mtoks(t); df.update(w | n)                       # count each token once per passage
    needed = set()
    for it in items:                                     # each gold page's 40 rarest tokens found in the corpus
        # Rare tokens (mostly specific numbers such as 1577) pinpoint the right passage far better than
        # common words; tokens absent from the corpus (df = 0) can't help and are skipped.
        it["rare"] = sorted((t for t in it["toks"] if df[t] > 0), key=lambda t: (df[t], t))[:40]
        needed.update(it["rare"])
    postings = defaultdict(list)                         # pass 2: postings for those tokens only
    for i, t in enumerate(texts):                                # token -> rows of passages containing it
        w, n = _mtoks(t)
        for tok in (w | n) & needed:
            postings[tok].append(i)
    _say(f"phase B: {len(ids):,} passages indexed | {len(df):,} distinct tokens | postings kept for {len(needed):,} rare tokens")
    for it in (items if TRACE else []):
        if it["qid"] == _tq:
            _say(f"{_tq}: 10 rarest of its {len(it['rare'])} candidate-finding tokens: {it['rare'][:10]}")

    # ---- PHASE C: label each evidence item
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    diag = []
    for it in items:
        # the evidence snippet itself; failing that, the full gold page (e.g. a page-level corpus)
        exact = next((sorted(by_text[k]) for k in it["exact_keys"] if k and by_text.get(k)), [])
        if exact:
            for cid in exact:
                qrels[it["qid"]][cid] = 1
            diag.append({"query_id": it["qid"], "financebench_id": qmap[it["qid"]], "evidence_idx": it["j"],
                         "doc_name": it["doc"], "page": it["page"], "n_matched": len(exact), "matched": exact,
                         "possible_duplicates": len(exact) - 1, "best_containment": 1.0, "method": "exact"})
            continue                                             # done with this item; no fuzzy matching
        # No exact match: candidates are the passages sharing the most rare tokens with the gold page.
        cand = Counter()                                         # passage row -> rare tokens shared
        for tok in it["rare"]:
            cand.update(postings[tok])
        best, picked = 0.0, []                           # best: highest containment seen (diagnostics)
        for i, _ in cand.most_common(n_candidates):
            T = _shingles(texts[i])
            if len(T) < min_shingles:                            # too short to judge reliably (~110 chars)
                continue
            _, tn = _mtoks(texts[i])
            contain = len(T & it["P"]) / len(T)                  # rule (a): share of the passage found on the gold page
            num_ok = len(tn & it["nums"]) / len(tn) if len(tn) >= 3 else 1.0   # rule (a): share of its numbers on the page
            overlap = len(T & it["E"])                           # rule (b): 10-grams shared with the evidence
            best = max(best, contain)
            member = contain >= member_min and num_ok >= num_min
            # rule (b): the passage covers >= half the evidence, or >= 80% of the passage is evidence
            if member and it["E"] and (overlap / len(it["E"]) >= 0.5 or overlap / len(T) >= 0.8):
                picked.append((ids[i], len(T & it["P"]) / len(it["P"])))   # (id, share of the gold page it covers)
        for cid, _ in picked:
            qrels[it["qid"]][cid] = 1
        full_copies = [c for c, cover in picked if cover >= 0.9]          # passages that are (nearly) the whole page
        diag.append({"query_id": it["qid"], "financebench_id": qmap[it["qid"]], "evidence_idx": it["j"],
                     "doc_name": it["doc"], "page": it["page"], "n_matched": len(picked),
                     "matched": [c for c, _ in picked], "possible_duplicates": max(0, len(full_copies) - 1),
                     "best_containment": round(best, 3), "method": "fuzzy"})
    _say(f"phase C: exact {sum(d['method'] == 'exact' for d in diag)} | fuzzy "
         f"{sum(d['method'] == 'fuzzy' and d['n_matched'] > 0 for d in diag)} | not found {sum(d['n_matched'] == 0 for d in diag)}")
    for d in (diag if TRACE else []):
        if d["query_id"] == _tq:
            _say(f"{_tq}: method {d['method']} -> labelled {d['matched']} (best containment {d['best_containment']})")
    return dict(qrels), diag


@_traced
def compare_qrels(official: dict, derived: dict) -> dict:
    # Per query: are the two answer keys the same set of passages, or do they at least share one?
    qs = sorted(set(official) | set(derived))
    same = sum(1 for q in qs if set(official.get(q, {})) == set(derived.get(q, {})))
    overlap = sum(1 for q in qs if set(official.get(q, {})) & set(derived.get(q, {})))
    return {"queries": len(qs), "identical": same, "any_overlap": overlap,
            "only_official": [q for q in qs if q in official and q not in derived],
            "only_derived": [q for q in qs if q in derived and q not in official]}


# =========================================================================== STEP 4: BM25 baseline and output files
_FY = re.compile(r"\b(FY|Q[1-4])\s*'?(\d{2,4})\b", re.I)    # "FY2018", "FY 22", "Q2'23" -> prefix + year


def _bm25_text(s: str) -> str:
    """Split filing names and fiscal periods so they can match: '3M_2018_10K' -> '3M 2018 10K',
    'FY2018' -> 'FY 2018'. Without this the titles never match a query word (underscores are word characters)."""
    return _FY.sub(r"\1 \2", s.replace("_", " "))


@_traced
def bm25_run(corpus: dict[str, dict], queries: dict[str, str], top_k: int = 100) -> dict[str, dict[str, float]]:
    """BM25 over title + text of every passage. Returns a run {query_id: {passage_id: score}}, top_k per query."""
    import bm25s  # pip install bm25s
    ids = list(corpus)                                           # row number i in the index <-> ids[i]
    # Lowercase word tokens with English stopwords removed, after splitting filing names and FY periods.
    tokens = bm25s.tokenize([_bm25_text(corpus[c]["title"] + " " + corpus[c]["text"]).strip() for c in ids],
                            stopwords="en", show_progress=False)
    model = bm25s.BM25()
    model.index(tokens, show_progress=False)                     # builds the term-weight matrix once
    _say(f"indexed {len(ids):,} passages, vocabulary {len(tokens.vocab):,} words; a title is indexed as "
         f"{_bm25_text(corpus[ids[0]]['title'])!r} (was {corpus[ids[0]]['title']!r})")
    qids = list(queries)
    # return_ids=False: queries as plain word lists, looked up in the corpus vocabulary
    qtok = bm25s.tokenize([_bm25_text(queries[q]) for q in qids], stopwords="en", return_ids=False, show_progress=False)
    if TRACE and _tq in queries:
        words = qtok[qids.index(_tq)]
        _say(f"{_tq} as BM25 sees it ({len(words)} words): {' '.join(words[:30])}{' ...' if len(words) > 30 else ''}")
    k = min(top_k, len(ids))                                     # can't return more passages than exist
    docs, scores = model.retrieve(qtok, k=k, show_progress=False)  # arrays [n_queries x k]: rows and scores
    # Back from row numbers to ids: {query_id: {passage_id: score}}
    return {q: {ids[int(d)]: float(s) for d, s in zip(docs[i], scores[i])} for i, q in enumerate(qids)}


@_traced
def write_submission(run: dict[str, dict[str, float]], path: str, top_k: int = 10) -> None:
    """Kaggle / FinanceRAG save_results format: header query_id,corpus_id; top_k rows per query by rank."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "corpus_id"])
        for q, scores in run.items():
            for d in _ranked(scores)[:top_k]:                    # best first; row order carries the rank
                w.writerow([q, d])


@_traced
def write_per_query(per_query: dict, path: str, qmap=None, fb_rows=None) -> None:
    # One CSV row per query: its FinanceBench id, type and filing (when known), and its key metrics.
    fb = {r["financebench_id"]: r for r in (fb_rows or [])}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "financebench_id", "question_type", "doc_name", "first_rel_rank",
                    "nDCG@10", "Recall@10", "Recall@100", "MRR@10"])
        for q, m in per_query.items():
            fid = (qmap or {}).get(q, "")
            r = fb.get(fid, {})
            # m is None when the run had no results for this query: write empty cells
            vals = ["" if m is None else m.get(k) for k in ("first_rel_rank", "nDCG@10", "Recall@10", "Recall@100", "MRR@10")]
            w.writerow([q, fid, r.get("question_type", ""), r.get("doc_name", "")] + vals)


@_traced
def by_question_type(per_query: dict, qmap: dict, fb_rows: list[dict], metric: str = "nDCG@10") -> dict:
    # Average one metric per FinanceBench question type: {"domain-relevant": (mean, n), ...}.
    # Queries the run skipped count as 0, like the "strict" average.
    qt = {r["financebench_id"]: r["question_type"] for r in fb_rows}
    groups = defaultdict(list)
    for q, m in per_query.items():
        groups[qt.get(qmap.get(q), "unmatched")].append(0.0 if m is None else m[metric])
    return {g: (sum(v) / len(v), len(v)) for g, v in sorted(groups.items())}


def answers_to_preds(answers_path: str, qmap: dict[str, str]) -> list[dict]:
    """FinanceRAG generate() output {"query_id","answer"} -> financebench_starter predictions.
    The query id is swapped for its FinanceBench id, so fs.score_preds() can grade the answer."""
    return [{"financebench_id": qmap[str(r["query_id"])], "answer": r["answer"]}
            for r in _read_jsonl(answers_path) if str(r["query_id"]) in qmap]


# =========================================================================== STEP 4b: more rankers and filing routing
# These rankers score EVERY passage for every query: a matrix S[query, passage]. Filing routing re-orders those
# scores, and the top_k of each row become the run that STEP 5 scores. bm25_run above is left exactly as it was,
# so the plain BM25 baseline gives the same numbers as before.
METHODS = ("bm25", "dense", "hybrid")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # small and fast on a laptop CPU; "BAAI/bge-base-en-v1.5" or "intfloat/e5-base-v2" also work


@_traced
def _bm25_matrix(corpus: dict[str, dict], queries: dict[str, str]):
    """BM25 score of every passage for every query, tokenised exactly like bm25_run: a [queries x passages] array."""
    import bm25s, numpy as np
    ids, qids = list(corpus), list(queries)
    tokens = bm25s.tokenize([_bm25_text(corpus[c]["title"] + " " + corpus[c]["text"]).strip() for c in ids],
                            stopwords="en", show_progress=False)
    model = bm25s.BM25()
    model.index(tokens, show_progress=False)
    qtok = bm25s.tokenize([_bm25_text(queries[q]) for q in qids], stopwords="en", return_ids=False, show_progress=False)
    S = np.zeros((len(qids), len(ids)), dtype="float32")
    for i, words in enumerate(qtok):
        known = [tokens.vocab[w] for w in words if w in tokens.vocab]      # words no passage uses add nothing
        if known:
            S[i] = model.get_scores(known)
    _say(f"BM25 scores for all {len(ids):,} passages x {len(qids)} queries (vocabulary {len(tokens.vocab):,} words)")
    return S


def _dense_prefixes(model_name: str) -> tuple[str, str]:
    # (query prefix, passage prefix) that a model's card asks for. Most models need none.
    m = model_name.lower()
    if re.search(r"(^|[/_-])e5-", m):                                  # E5 family: "query: " and "passage: "
        return "query: ", "passage: "
    if "bge-" in m and "-en" in m:                                     # English BGE: an instruction on queries only
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


@_traced
def _dense_matrix(corpus: dict[str, dict], queries: dict[str, str], model_name: str = DENSE_MODEL, batch_size: int = 32):
    """Cosine similarity of every query with every passage, from a sentence-transformers embedding model.
    Needs `pip install sentence-transformers`; the model downloads from Hugging Face on first use, then is cached.
    model_name can also be a local folder holding a downloaded model.
    Passages longer than the model's limit (512 tokens for BGE) are cut at that limit."""
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("dense and hybrid need sentence-transformers: pip install sentence-transformers") from None
    ids, qids = list(corpus), list(queries)
    qp, pp = _dense_prefixes(model_name)
    model = SentenceTransformer(model_name)
    # The filing name goes in front of each passage ("3M 2018 10K"), so the company and year are embedded too.
    P = model.encode([pp + _bm25_text(corpus[c]["title"]) + "\n" + corpus[c]["text"] for c in ids], batch_size=batch_size,
                     show_progress_bar=len(ids) > 500, convert_to_numpy=True, normalize_embeddings=True)
    Q = model.encode([qp + queries[q] for q in qids], batch_size=batch_size, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
    _say(f"{model_name}: one {P.shape[1]}-number vector per text; query prefix {qp!r}, passage prefix {pp!r}")
    return np.asarray(Q @ P.T, dtype="float32")                       # unit-length vectors: dot product = cosine


def _rrf(*mats, k: int = 60):
    """Reciprocal rank fusion: each ranker adds 1 / (k + rank) for every passage. Ranks are added, not raw
    scores, so BM25 scores (0 to about 30) and cosine similarities (-1 to 1) combine without rescaling."""
    import numpy as np
    out = np.zeros(mats[0].shape)
    for S in mats:
        ranks = np.argsort(np.argsort(-S, axis=1, kind="stable"), axis=1, kind="stable") + 1   # 1 = best, per query
        out += 1.0 / (k + ranks)
    return out


# ---- filing routing: the filing a question names goes first
# Short names and tickers people write instead of the company name in the titles. jnj, jpm, mgm and amex come from
# your 150 questions (19 of them use one); add the names your own users type.
ALIASES = {"jnj": "johnsonjohnson", "jpm": "jpmorgan", "mgm": "mgmresorts", "amex": "americanexpress"}
_TITLE = re.compile(r"^(?P<co>.+?)_(?P<year>\d{4})(?:Q(?P<q>[1-4]))?_")    # "3M_2023Q2_10Q": company, year, quarter
_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _filing(title: str):
    # "3M_2023Q2_10Q" -> ("3m", 2023, 2); quarter 0 for annual filings; None if the title isn't a filing name.
    m = _TITLE.match(title or "")
    return (re.sub(r"[^a-z0-9]", "", m["co"].lower()), int(m["year"]), int(m["q"] or 0)) if m else None


def _hints(question: str, companies: set) -> tuple[set, set, set]:
    """(companies, years, quarters) a question names. A company is found by joining 1-4 neighbouring words, so
    "American Express" matches the title company AMERICANEXPRESS; ALIASES adds short names such as JPM."""
    ql = question.lower()
    w = re.findall(r"[a-z0-9]+", ql)
    runs = {"".join(w[i:i + n]) for n in (1, 2, 3, 4) for i in range(len(w) - n + 1)}
    cos = {r for r in runs if r in companies} | {ALIASES[r] for r in runs if ALIASES.get(r) in companies}
    years = {int(y) for y in re.findall(r"\b(?:fy\s*'?)?((?:19|20)\d{2})\b", ql)}
    years |= {2000 + int(y) for y in re.findall(r"\bfy\s*'?(\d{2})\b", ql)}                  # "FY22" -> 2022
    quarters = {int(x) for x in re.findall(r"\bq([1-4])\b", ql)}
    quarters |= {_QUARTER_WORDS[x] for x in re.findall(r"\b(first|second|third|fourth) quarter\b", ql)}
    return cos, years, quarters


@_traced
def route_scores(S, corpus: dict[str, dict], queries: dict[str, str]):
    """Filing routing. A ranker's scores are squeezed into 0..1 per query, then passages of the company the question
    names get +16, +8 more if the filing's year is the latest year named (a filing also reports the year before),
    +4 for another named year, and +2 for a named quarter. So the named filing comes first and the ranker's own
    order holds inside each group. Questions that name no company keep the ranker's order."""
    import numpy as np
    ids, qids = list(corpus), list(queries)
    fil = [_filing(corpus[c]["title"]) for c in ids]
    companies = {f[0] for f in fil if f}
    if not companies:
        print("note: no passage title looks like a filing name (COMPANY_YEAR_TYPE), so routing changes nothing")
    co = np.array([f[0] if f else "" for f in fil])
    yr = np.array([f[1] if f else 0 for f in fil])
    qt = np.array([f[2] if f else 0 for f in fil])
    R = np.zeros(S.shape)
    routed = 0
    for i, q in enumerate(qids):
        s = S[i].astype("float64")
        R[i] = (s - s.min()) / (s.max() - s.min() + 1e-12)            # the ranker's order, as numbers in 0..1
        cos, years, quarters = _hints(queries[q], companies)
        if not cos:
            continue
        routed += 1
        same = np.isin(co, list(cos))
        bonus = 16.0 * same
        if years:
            latest = max(years)
            bonus += same * (8.0 * (yr == latest) + 4.0 * ((yr != latest) & np.isin(yr, list(years))))
        if quarters:
            bonus += same * 2.0 * np.isin(qt, list(quarters))
        R[i] += bonus
        if q == _tq:
            first = sorted({corpus[ids[j]]["title"] for j in np.flatnonzero(bonus == bonus.max())})
            _say(f"{q}: names company {sorted(cos)}, years {sorted(years) or '-'}, quarters {sorted(quarters) or '-'}"
                 f" -> first: {', '.join(first)}")
    _say(f"routed {routed} of {len(qids)} questions to the filings of a company they name")
    return R


@_traced
def retrieve(corpus: dict[str, dict], queries: dict[str, str], method: str = "bm25", route: bool = False,
             model: str = DENSE_MODEL, top_k: int = 100, _cache: dict | None = None) -> dict[str, dict[str, float]]:
    """Rank passages for every query; returns a run {query_id: {passage_id: score}} with top_k per query.
      method  bm25 (the baseline), dense (an embedding model) or hybrid (BM25 and dense fused by rank)
      route   put the filing each question names first (filing routing)"""
    import numpy as np
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: use one of {', '.join(METHODS)}")
    if method == "bm25" and not route:
        return bm25_run(corpus, queries, top_k)                        # the original baseline, unchanged
    cache = {} if _cache is None else _cache                           # compare() reuses the matrices across methods
    if method in ("bm25", "hybrid") and "bm25" not in cache:
        cache["bm25"] = _bm25_matrix(corpus, queries)
    if method in ("dense", "hybrid") and ("dense", model) not in cache:
        cache[("dense", model)] = _dense_matrix(corpus, queries, model)
    if method == "hybrid":
        S = _rrf(cache["bm25"], cache[("dense", model)])
        _say("hybrid: each passage scores 1/(60 + its BM25 rank) + 1/(60 + its dense rank)")
    else:
        S = cache["bm25"] if method == "bm25" else cache[("dense", model)]
    if route:
        S = route_scores(S, corpus, queries)
    ids, k = list(corpus), min(top_k, len(corpus))
    return {q: {ids[j]: float(S[i, j]) for j in np.argpartition(-S[i], k - 1)[:k]} for i, q in enumerate(queries)}


# =========================================================================== STEP 5: score a run (run_eval)
@_traced
def _fb_rows(fb_dir: str | None) -> list[dict]:
    # Load the 150 FinanceBench questions with financebench_starter.load(). The import happens here, not
    # at the top, so this file works without financebench_starter.py when --fb isn't used.
    if not fb_dir:
        return []
    # Look next to this file. When this code was pasted into a notebook cell there is no __file__,
    # so look in the notebook's folder instead.
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import financebench_starter as fs
    except ImportError:
        # financebench_starter.py itself pasted into an earlier cell: its load() is already defined here.
        if callable(globals().get("load")) and callable(globals().get("score_preds")):
            return globals()["load"](fb_dir)
        raise ImportError(f"financebench_starter.py not found: put it in {here}, or paste it into a cell above") from None
    return fs.load(fb_dir)


@_traced
def run_eval(corpus: dict, queries: dict, run: dict, fb_dir: str | None = None, qrels_path: str | None = None,
             out: str | None = None, title: str = "run") -> dict:
    """Score a run against official qrels (if given) and qrels derived from FinanceBench (if fb_dir given).
    Returns {"derived": {...}, "official": {...}}; each holds "summary", "per_query" and "qrels"
    ("derived" also holds "diag" and "qmap")."""
    print(f"corpus: {len(corpus):,} entries | queries: {len(queries)} | run covers {len(run)} queries")
    _say(f"plan: 5a derived answer key {'yes' if fb_dir else 'no (needs --fb)'} | 5b official answer key "
         f"{'yes' if qrels_path else 'no (no qrels file)'} | 5c write files {'yes' if out else 'no (needs --out)'}")
    fb_rows = _fb_rows(fb_dir)
    qmap, results = {}, {}
    if out:
        os.makedirs(out, exist_ok=True)
    if fb_rows:
        # ---- 5a. answer key derived from FinanceBench (needs --fb)
        qmap, info = match_queries(queries, fb_rows)                 # query id -> financebench_id
        print(f"queries matched to FinanceBench: {info['matched']}/{len(queries)}"
              + (f" | unmatched ids: {info['unmatched'][:5]}" if info["unmatched"] else ""))
        derived, diag = derive_qrels(corpus, fb_rows, qmap)          # answer key for every matched query
        none = [d for d in diag if d["n_matched"] == 0]              # evidence items with no passage found
        dups = sum(d["possible_duplicates"] for d in diag)
        print(f"derived qrels: {len(derived)} queries | gold evidence pages {len(diag)}: "
              f"{len(diag) - len(none)} found in corpus, {len(none)} not found, {dups} with identical copies")
        for d in none[:10]:
            print(f"   not found: {d['financebench_id']} {d['doc_name']} page {d['page']} "
                  f"(best containment {d['best_containment']})")
        s, pq = evaluate(derived, run)                               # metrics vs the derived answer key
        print_summary(s, f"{title} vs qrels derived from FinanceBench evidence")
        for g, (v, n) in by_question_type(pq, qmap, fb_rows).items():
            print(f"   nDCG@10 {g:18s} {v:.4f}  (n={n})")
        results["derived"] = {"summary": s, "per_query": pq, "qrels": derived, "diag": diag, "qmap": qmap}
        if out:                                                      # 5c. files for the derived key
            write_per_query(pq, os.path.join(out, "per_query_derived.csv"), qmap, fb_rows)
            with open(os.path.join(out, "qrels_derived.tsv"), "w", encoding="utf-8") as f:
                f.write("query_id\tcorpus_id\tscore\n")
                for q, rels in derived.items():
                    for c, g in rels.items():
                        f.write(f"{q}\t{c}\t{g}\n")
    if qrels_path:
        # ---- 5b. the official answer key file (found under --data-root or given with --qrels)
        official = load_qrels(qrels_path)
        s, pq = evaluate(official, run)
        print_summary(s, f"{title} vs official qrels ({os.path.basename(qrels_path)})")
        results["official"] = {"summary": s, "per_query": pq, "qrels": official}
        if "derived" in results:                                     # how well the two answer keys agree
            c = compare_qrels(official, results["derived"]["qrels"])
            print(f"   official vs derived qrels: identical for {c['identical']}/{c['queries']} queries, "
                  f"overlap for {c['any_overlap']} | only in official {len(c['only_official'])}, "
                  f"only in derived {len(c['only_derived'])}")
        if out:                                                      # 5c. file for the official key
            write_per_query(pq, os.path.join(out, "per_query_official.csv"), qmap, fb_rows)
    if not fb_rows and not qrels_path:
        print("nothing to score against: pass --qrels and/or --fb")
    return results


@_traced
def compare(corpus: dict, queries: dict, methods: list[str], fb_dir: str | None = None, qrels_path: str | None = None,
            model: str = DENSE_MODEL, top_k: int = 100, out: str | None = None) -> dict:
    """Score several rankers against the same answer key and print one table (also out/compare.csv with --out).
    methods: names such as "bm25", "bm25+route", "dense", "hybrid+route". Each row's difference from bm25 comes with
    a 95% interval from resampling the questions (paired bootstrap): an interval that excludes 0 is a real change.
    Returns {method: {"derived" / "official": (summary, per_query)}}."""
    import numpy as np
    names = [m.strip() for m in methods if m.strip()]
    for name in names:                                                 # check every name before doing any work
        method, _, extra = name.partition("+")
        if method not in METHODS or extra not in ("", "route"):
            raise ValueError(f"unknown method {name!r}: use bm25, dense or hybrid, optionally followed by +route")
    keys = {}                                                          # the answer keys to score against
    fb_rows = _fb_rows(fb_dir)
    if fb_rows:
        qmap, _ = match_queries(queries, fb_rows)
        keys["derived"] = derive_qrels(corpus, fb_rows, qmap)[0]
    if qrels_path:
        keys["official"] = load_qrels(qrels_path)
    if not keys:
        raise ValueError("compare needs an answer key: pass --fb and/or --qrels")
    main_key = "derived" if "derived" in keys else "official"
    cache, evals = {}, {}
    if any(n.partition("+")[0] != "bm25" for n in names):             # dense and hybrid share one embedding run
        try:
            cache[("dense", model)] = _dense_matrix(corpus, queries, model)
        except Exception as e:                                         # not installed, or the model can't download
            print(f"   dense and hybrid skipped: {type(e).__name__}: {e}")
            names = [n for n in names if n.partition("+")[0] == "bm25"]
    for name in names:
        method, _, extra = name.partition("+")
        run = retrieve(corpus, queries, method, extra == "route", model, top_k, cache)
        evals[name] = {k: evaluate(qrels, run) for k, qrels in keys.items()}
    per_q = lambda name, m: np.array([0.0 if v is None else v[m] for v in evals[name][main_key][1].values()])
    base = per_q("bm25", "nDCG@10") if "bm25" in evals else None
    rng = np.random.default_rng(0)
    off = "official" in keys and main_key != "official"
    print(f"\n=== compare: {len(base) if base is not None else '?'} questions in the {main_key} answer key"
          + (f"; 'official' = the {len(keys['official'])} questions of the qrels file" if off else ""))
    print(f"   {'method':14s} {'nDCG@10':>7s}  {'vs bm25 [95% interval]':>25s}" + (f"  {'official':>8s}" if off else "")
          + f"  {'Recall@10':>9s}  {'MRR@10':>6s}")
    rows = []
    for name in evals:
        x = per_q(name, "nDCG@10")
        lo = hi = d = None
        if base is not None and name != "bm25":
            d = x - base                                               # per-question gain over bm25
            lo, hi = np.percentile(d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1), [2.5, 97.5])
        o = [evals[name]["official"][0]["nDCG@10"]["official"]] if off else []
        r10, mrr = per_q(name, "Recall@10").mean(), per_q(name, "MRR@10").mean()
        rows.append([name, x.mean()] + ([d.mean(), lo, hi] if d is not None else [None] * 3) + o + [r10, mrr])
        print(f"   {name:14s} {x.mean():7.4f}  "
              + (f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]".rjust(25) if d is not None else f"{'-':>25s}")
              + "".join(f"  {v:8.4f}" for v in o) + f"  {r10:9.4f}  {mrr:6.4f}")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "compare.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", f"nDCG@10 ({main_key})", "vs bm25", "95% low", "95% high"]
                       + (["nDCG@10 (official qrels)"] if off else []) + ["Recall@10", "MRR@10"])
            w.writerows([[v if isinstance(v, str) or v is None else round(float(v), 4) for v in r] for r in rows])
        print(f"wrote {os.path.join(out, 'compare.csv')}")
    return evals


# =========================================================================== STEP 6: command line / notebook entry point
def main(argv=None, trace=None) -> dict:
    """Entry point for both the command line and notebooks.
    argv=None: in a notebook, runs NOTEBOOK_ARGS with the trace on; on the command line, reads the real arguments.
    In a notebook you can also pass a list, e.g. main(["find", "--data-root", r"D:\\x"]).
    trace=True/False turns the step-by-step trace on or off (default: on for main() in a notebook, else --trace).
    Returns (useful in a notebook):
      res["run"]                   {query_id: {passage_id: score}}
      res["derived"]["summary"]    metrics vs the answer key built from FinanceBench   (with --fb)
      res["derived"]["per_query"]  metrics for each query;  res["derived"]["qmap"]  query id -> financebench_id
      res["official"]["summary"]   metrics vs the qrels file                            (if one was found/given)"""
    global TRACE, _tq
    _tq = None
    if argv is None and _in_notebook():                  # a notebook cell or interactive Python
        argv = list(NOTEBOOK_ARGS)        # sys.argv there holds the kernel's own arguments, not ours
        print("main() called with no arguments in a notebook, so it runs NOTEBOOK_ARGS:\n   " + " ".join(argv))
        if trace is None:
            trace = True
    # ---- 6.1 read the arguments
    ap = argparse.ArgumentParser(description="FinanceRAG adapter for the FinanceBench harness")
    ap.add_argument("cmd", choices=["baseline", "eval", "find", "compare"])
    src = ap.add_argument_group("where the competition files are (pick one)")
    src.add_argument("--data-root", help=r"folder holding the competition data, e.g. D:\Fin_Rag\finrag_data")
    src.add_argument("--task-dir", help="folder with corpus.jsonl and queries.jsonl")
    src.add_argument("--corpus-file", help="explicit corpus .jsonl (with --queries-file), if auto-detection can't find it")
    src.add_argument("--queries-file", help="explicit queries .jsonl")
    ap.add_argument("--task", default="FinanceBench", help="task name used with --data-root (default FinanceBench)")
    ap.add_argument("--fb", help="financebench_starter data folder, e.g. financebench (derived qrels + per-type results)")
    ap.add_argument("--qrels", help="official qrels TSV/CSV; found automatically under --data-root if present")
    ap.add_argument("--run", help="eval: your results file (Kaggle results.csv, TREC run, or JSONL)")
    ap.add_argument("--out", help="output folder for submission, per-query CSVs and derived qrels")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--trace", action="store_true", help="print each step of the code flow and the data it passes on")
    rk = ap.add_argument_group("rankers (baseline and compare)")
    rk.add_argument("--method", default="bm25", choices=METHODS, help="baseline: bm25 (default), dense or hybrid")
    rk.add_argument("--route", action="store_true", help="baseline: put the filing each question names first")
    rk.add_argument("--model", default=DENSE_MODEL, help=f"embedding model for dense and hybrid (default {DENSE_MODEL})")
    rk.add_argument("--methods", default="bm25,bm25+route,dense,dense+route,hybrid,hybrid+route",
                    help="compare: comma-separated rankers")
    a = ap.parse_args(argv)
    TRACE = a.trace = a.trace if trace is None else bool(trace)
    _say(f"== STEP 6.1 read the arguments: {vars(a)}")
    # ---- 6.2 check the arguments make sense together
    explicit = bool(a.corpus_file and a.queries_file)
    if sum([bool(a.data_root), bool(a.task_dir), explicit]) != 1:      # exactly one way of locating files
        ap.error("give one of --data-root, --task-dir, or --corpus-file with --queries-file")
    if a.cmd == "eval" and not a.run:
        ap.error("eval needs --run")
    _say("== STEP 6.2 the arguments fit together")
    # ---- 6.3 locate the files (STEP 1)
    _say("== STEP 6.3 locate the files (STEP 1)" + ("" if a.data_root else ": paths given, no search"))
    if a.data_root:
        paths = find_task_files(a.data_root, a.task)
    elif a.task_dir:
        paths = {"corpus": os.path.join(a.task_dir, "corpus.jsonl"),
                 "queries": os.path.join(a.task_dir, "queries.jsonl"), "qrels": None}
    else:
        paths = {"corpus": a.corpus_file, "queries": a.queries_file, "qrels": None}
    for kind in ("corpus", "queries"):
        if not os.path.isfile(paths[kind]):
            raise FileNotFoundError(f"{kind} file not found: {paths[kind]!r}")
    qrels_path = a.qrels or paths["qrels"]                       # an explicit --qrels beats the one found
    print(f"corpus : {paths['corpus']}\nqueries: {paths['queries']}\nqrels  : {qrels_path or 'none found'}")
    if a.cmd == "find":                                          # "find" stops here: just report the paths
        _say("== find stops here: it only reports the paths")
        return {"paths": paths, "qrels": qrels_path}
    # ---- 6.4 load the files (STEP 1)
    _say("== STEP 6.4 load the files (STEP 1)")
    corpus, queries = load_corpus(paths["corpus"]), load_queries(paths["queries"])
    _tq = TRACE_QUERY if TRACE_QUERY in queries else next(iter(queries), None)   # the query to follow
    if _tq:
        _say(f"following query {_tq} through every step: {' '.join(queries[_tq].split())}")
    if a.cmd == "compare":                                       # several rankers, one table, then stop
        _say("== STEP 6.5 compare rankers (STEP 4b for each ranking, STEP 2 for its scores)")
        results = compare(corpus, queries, a.methods.split(","), a.fb, qrels_path, a.model, a.top_k, a.out)
        _say(f"== done: res = {_short(results)}")
        return results
    # ---- 6.5 get a ranking: compute one (baseline, STEP 4/4b) or read yours (eval, STEP 1)
    tag = a.method + ("_routed" if a.route else "")              # bm25, bm25_routed, dense, hybrid_routed, ...
    _say("== STEP 6.5 get a ranking: " + (f"{tag} (STEP 4/4b)" if a.cmd == "baseline" else "read your run file (STEP 1)"))
    if a.cmd == "baseline":
        run = retrieve(corpus, queries, a.method, a.route, a.model, a.top_k)
        title = "BM25 baseline" if tag == "bm25" else f"{a.method}{' + filing routing' if a.route else ''}"
        if a.out:
            write_submission(run, os.path.join(a.out, f"submission_{tag}.csv"))
            print(f"wrote {os.path.join(a.out, f'submission_{tag}.csv')} (Kaggle format, top 10 per query)")
    else:
        run = load_run(a.run)
        title = os.path.basename(a.run)
    if TRACE and _tq in run:
        _say(f"{_tq}: top 5 of the run (rank, passage, filing, score)")
        for i, d in enumerate(_ranked(run[_tq])[:5], 1):
            _say(f"   {i}. {d}  {corpus.get(d, {}).get('title', '?'):28s} {run[_tq][d]:8.3f}")
    # ---- 6.6 score it (STEP 5) and hand everything back
    _say("== STEP 6.6 score the run (STEP 5)")
    results = run_eval(corpus, queries, run, a.fb, qrels_path, a.out, title)
    results["run"] = run
    _say(f"== done: res = {_short(results)}")
    return results


if __name__ == "__main__" and not _in_notebook():   # pasted into a notebook: run res = main() in the next cell
    main()
