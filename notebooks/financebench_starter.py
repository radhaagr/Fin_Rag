"""
FinanceBench starter harness (open-source 150-question sample)
==============================================================
Data: github.com/patronus-ai/financebench  (licence CC-BY-NC-4.0 -> non-commercial use only)

What this gives you
  setup     download questions, document metadata, human-labelled baselines (+ PDFs with --pdfs)
  load()    150 questions joined with doc metadata; gold evidence as (doc_name, page) pairs
  pages()   page-level text from a filing, 0-INDEXED (matches evidence_page_num in the current JSONL;
            the paper and the 2023 CSV used 1-indexed pages)
  metrics   doc recall@k, page recall@k for your retriever
  scoring   deterministic numeric scorer for metrics-generated questions; LLM-judge hook for the rest;
            3-way labels like the paper: correct / incorrect / refusal
  report    accuracy per question_type with Wilson 95% CIs, with and without contested items
  calibrate agreement (Cohen's kappa) of any judge vs the 2,400 human labels released in /results

Usage
  python financebench_starter.py setup    --dir fb [--pdfs] [--corpus sample|all|all+unlabelled]
  python financebench_starter.py selftest --dir fb
  python financebench_starter.py demo     --dir fb [--pred preds.jsonl] [--config <released run>]
  python financebench_starter.py score    --dir fb --pred preds.jsonl
      preds.jsonl is YOUR system's output (demo writes one from a released baseline), one line per question:
        {"financebench_id": "...", "answer": "...", "retrieved": [["DOC_NAME", page0], ...]}
      retrieved is optional (ranked, pages 0-indexed); questions with no line count as wrong

Notebook (don't call argparse with Jupyter's own arguments; pass a list or use the functions)
    import financebench_starter as fs
    fs.main(["demo", "--dir", "financebench"])                       # writes preds.jsonl
    labels = fs.main(["score", "--dir", "financebench", "--pred", "preds.jsonl"])
    # or: rows = fs.load("financebench"); preds = fs.read_preds("preds.jsonl")
    #     labels = fs.score_preds(rows, preds); fs.report(rows, labels, "my run"); fs.retrieval_report(rows, preds)

Free-text questions (domain-relevant, novel-generated) need an LLM judge; plug in any client:
    call_llm = lambda prompt: my_client_complete(prompt)          # str -> str
    labels = fs.score_preds(rows, preds, call_llm)
    fs.calibrate("financebench", lambda row, ans: fs.score_one(row, ans, call_llm))   # judge vs human labels

Numeric scorer vs the released human labels (800 metrics-question completions):
3-way agreement 92.4%, correct-vs-not 93.9%, Cohen's kappa 0.88. Residual disagreement is mostly
inconsistent human leniency on 0.1-0.2pt rounding differences.
"""
from __future__ import annotations
import argparse, contextlib, json, math, os, re, shutil, sys, time
import urllib.error, urllib.request
from collections import Counter, defaultdict

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
RESULT_FILES = [f"{m}_{e}" for m, e in [
    ("claude-2", "inContext"), ("claude-2", "inContext_reverse"),
    ("gpt-4-1106-preview", "closedBook"), ("gpt-4-1106-preview", "inContext"),
    ("gpt-4-1106-preview", "inContext_reverse"), ("gpt-4-1106-preview", "oracle"),
    ("gpt-4-1106-preview", "oracle_reverse"), ("gpt-4-1106-preview", "sharedStore"),
    ("gpt-4-1106-preview", "singleStore"), ("gpt-4", "closedBook"), ("gpt-4", "oracle"),
    ("gpt-4", "oracle_reverse"), ("gpt-4", "sharedStore"), ("gpt-4", "singleStore"),
    ("llama2", "sharedStore"), ("llama2", "singleStore")]]

# Items flagged by two independent signals: all four oracle configs failed them in the
# released human labels AND VectifyAI's human review labelled them benchmark-error /
# multiple-valid-answers. Report scores with and without them.
CONTESTED = {"financebench_id_00005", "financebench_id_00070", "financebench_id_00080",
             "financebench_id_00283", "financebench_id_00499", "financebench_id_00540",
             "financebench_id_01911"}
# 00070's gold flipped Yes->No in Sep 2024; its human labels in /results predate the fix.
STALE_HUMAN_LABELS = {"financebench_id_00070"}
# PDFs in the repo's /pdfs with no row in the metadata file (repo tree at commit cc39aeb, Dec 2024).
# BOSTONPROPERTIES is not one of the 40 benchmark companies.
UNLABELLED_PDFS = ["ADOBE_2022Q2_10Q", "AMCOR_2022_8K_2022-04-26", "AMCOR_2022_8K_2022-07-01",
                   "BLOCK_2022_8K_dated-2022-01-28", "BOSTONPROPERTIES_2015_10K", "FOOTLOCKER_2022_10Q",
                   "ULTABEAUTY_2023Q1_EARNINGS", "ULTABEAUTY_2023Q2_EARNINGS"]


# --------------------------------------------------------------------------- data
def _complete(path: str, ext: str) -> bool:
    """Cheap integrity check, so a truncated file (e.g. from an interrupted run) is re-downloaded.
    PDF: starts with %PDF- and has %%EOF in the last 2 KB. JSONL: the last line parses."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if ext == ".pdf":
        with open(path, "rb") as f:
            head = f.read(5)
            f.seek(max(0, os.path.getsize(path) - 2048))
            return head == b"%PDF-" and b"%%EOF" in f.read()
    if ext == ".jsonl":
        with open(path, "rb") as f:
            lines = f.read().rstrip().splitlines()
        try:
            json.loads(lines[-1])
            return True
        except (ValueError, IndexError):
            return False
    return True


def _fetch(url: str, dest: str, retries: int = 3, timeout: float = 60) -> str:
    """Download url -> dest. Returns "cached" or "downloaded".
    - skips files that already pass the integrity check
    - writes to dest.part, verifies size + integrity, then atomically renames: dest is never partial
    - retries network errors / 5xx / 429 with backoff (2s, 4s); fails fast on other 4xx such as 404"""
    ext = os.path.splitext(dest)[1]
    if _complete(dest, ext):
        return "cached"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp, last_err = dest + ".part", None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                expected = resp.headers.get("Content-Length")
                shutil.copyfileobj(resp, out, 1 << 20)
            size = os.path.getsize(tmp)
            if expected is not None and size != int(expected):
                raise OSError(f"truncated download: {size} of {expected} bytes")
            if not _complete(tmp, ext):
                raise OSError("downloaded file failed integrity check")
            os.replace(tmp, dest)
            return "downloaded"
        except urllib.error.HTTPError as e:   # subclass of URLError/OSError, so must come first
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp)
            if 400 <= e.code < 500 and e.code != 429:
                raise                          # permanent, e.g. 404: retrying won't help
            last_err = e
        except OSError as e:                  # URLError, timeouts, truncation, failed check
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp)
            last_err = e
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last_err}")


def load_meta(d: str) -> dict[str, dict]:
    """doc_name -> filing metadata. The file has 361 rows but 360 unique names: FOOTLOCKER_2023_annualreport
    appears twice (doc_period 2023 and 2022). The first row is kept and the conflict is reported once; the
    kept row matches the report itself (fiscal year ended 28 Jan 2023 -> FY2023 under the dataset's convention)."""
    meta: dict[str, dict] = {}
    with open(f"{d}/data/financebench_document_information.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            prev = meta.setdefault(m["doc_name"], m)
            if prev is not m and prev != m and m["doc_name"] not in _WARNED:
                _WARNED.add(m["doc_name"])
                diff = {k: (prev.get(k), m.get(k)) for k in m if prev.get(k) != m.get(k)}
                print(f"note: metadata lists {m['doc_name']} twice; kept the first row, ignored {diff}", file=sys.stderr)
    return meta


_WARNED: set[str] = set()   # report each conflicting duplicate once per run


def corpus_docs(d: str, corpus: str = "sample") -> list[str]:
    """Filings that make up the retrieval corpus.
    sample         - the 84 filings the 150 questions use (the reference notebook's 'shared store')
    all            - all 360 filings in the metadata file (the paper's shared-store setting)
    all+unlabelled - plus the 8 repo PDFs with no metadata row (distractors)"""
    if corpus == "sample":
        return sorted({r["doc_name"] for r in load(d)})
    if corpus not in ("all", "all+unlabelled"):
        raise ValueError(f"corpus must be sample, all or all+unlabelled, not {corpus!r}")
    docs = sorted(load_meta(d))
    return docs + UNLABELLED_PDFS if corpus == "all+unlabelled" else docs


def setup(d: str, pdfs: bool = False, corpus: str = "sample") -> None:
    stats = Counter()
    for f in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
        stats[_fetch(f"{RAW}/data/{f}", f"{d}/data/{f}")] += 1      # failures here abort: nothing works without them
    for f in RESULT_FILES:
        stats[_fetch(f"{RAW}/results/{f}.jsonl", f"{d}/results/{f}.jsonl")] += 1
    failed = []
    if pdfs:
        docs = corpus_docs(d, corpus)
        for i, doc in enumerate(docs, 1):
            try:
                stats[_fetch(f"{RAW}/pdfs/{doc}.pdf", f"{d}/pdfs/{doc}.pdf")] += 1
            except (RuntimeError, urllib.error.HTTPError) as e:     # keep going; report at the end
                failed.append((doc, str(e)))
            if i % 50 == 0 or i == len(docs):
                print(f"  pdfs {i}/{len(docs)}")
    print(f"setup done: {d} | downloaded {stats['downloaded']}, already cached {stats['cached']}, failed {len(failed)}")
    if failed:
        raise RuntimeError("some PDFs failed (rerun setup to retry only these): " + "; ".join(f"{n}: {e}" for n, e in failed))


def load(d: str) -> list[dict]:
    meta = load_meta(d)
    rows = []
    with open(f"{d}/data/financebench_open_source.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = meta.get(r["doc_name"])
            if m is None:
                raise KeyError(f"{r['financebench_id']}: doc_name {r['doc_name']!r} not in the metadata file")
            r.update({k: m[k] for k in ("gics_sector", "doc_type", "doc_period", "doc_link")})
            r["gold_pages"] = sorted({(e["doc_name"], e["evidence_page_num"]) for e in r["evidence"]})
            rows.append(r)
    return rows


def iter_pages(pdf_path: str, sort: bool = True):
    """Yield (page_index, text), 0-indexed, one page at a time and closing the file afterwards, so you can
    index a 500-page filing without holding all of it in memory.
    sort=True orders text top-to-bottom, left-to-right (closer to reading order than the PDF's internal order).
    Tables still come out as lines: PyMuPDF's find_tables() fragmented 3M's 2018 cash-flow statement in testing,
    so use a layout-aware parser if you need table structure."""
    import pymupdf  # pip install pymupdf
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            yield i, page.get_text(sort=sort)


def pages(pdf_path: str, sort: bool = True) -> list[str]:
    """All page texts, 0-indexed: pages(p)[evidence_page_num] is the gold page."""
    return [text for _, text in iter_pages(pdf_path, sort)]


# --------------------------------------------------------------------------- retrieval metrics
def doc_recall_at_k(retrieved: list, gold_doc: str, k: int) -> float:
    return float(any(doc == gold_doc for doc, _ in retrieved[:k]))


def page_recall_at_k(retrieved: list, gold_pages: list, k: int) -> float:
    top = {tuple(x) for x in retrieved[:k]}
    return sum(tuple(g) in top for g in gold_pages) / len(gold_pages)


# --------------------------------------------------------------------------- numeric scoring
_NUM = re.compile(r"(\()?\s*([-−])?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(\))?\s*(%|percent)?", re.I)
_REFUSAL = re.compile(
    r"(cannot|can't|unable to|not able to|i'm sorry|i am sorry|as an ai|would need|"
    r"(?:does|do|did)(?: not|n't) (?:provide|contain|include|mention|specify|have|give|show)|"
    r"not (?:provided|available|included|mentioned|specified|possible)|insufficient|"
    r"no (?:specific )?(?:information|data|details)|if you (?:can )?provide|please (?:obtain|find|refer|consult|check)|"
    r"you (?:would|will) need to)", re.I)
_FINAL = re.compile(r"(final answer|answer is|therefore|thus|in summary|so,|approximately|equals|=)", re.I)


def parse_numbers(text: str) -> list[tuple[float, bool, int]]:
    """-> [(value, is_percent, n_decimals)]"""
    out = []
    for m in _NUM.finditer(str(text)):
        paren, minus, digits, close, unit = m.groups()
        try:
            v = float(digits.replace(",", ""))
        except ValueError:
            continue
        if minus or (paren and close):
            v = -v
        dec = len(digits.split(".")[1]) if "." in digits else 0
        out.append((v, bool(unit), dec))
    return out


def _round_match(p: float, pd: int, g: float, gd: int) -> bool:
    """Equal at the gold's precision, or the prediction is the gold rounded more coarsely."""
    if abs(p - g) <= 1e-9 * max(1.0, abs(g)):
        return True
    if round(p, gd) == round(g, gd):
        return True
    return pd < gd and round(g, pd) == round(p, pd)


def numeric_match(pred: str, gold: str) -> bool:
    """True if a number in the prediction's conclusion equals gold at gold precision.
    Handles $, commas, (negatives), % vs fraction, and millions<->billions/thousands shifts.
    Calibrated against the paper's human labels (strict on real differences, lenient on formatting)."""
    gnums = parse_numbers(gold)
    if not gnums:
        return False
    g, g_pct, _ = gnums[0]
    # Currency golds are padded ("$382.00" = rounded to whole millions): use significant decimals.
    gdig = re.search(r"\d[\d,]*(?:\.\d+)?", str(gold)).group(0)
    gd = len(gdig.split(".")[1].rstrip("0")) if "." in gdig else 0
    text = str(pred)
    m = list(_FINAL.finditer(text))
    cands = parse_numbers(text[m[-1].start():]) if m else []
    cands += parse_numbers(text[-400:])
    scaled = bool(re.search(r"billion|million|thousand", text, re.I))
    for v, v_pct, vd in cands:
        alts = [(v, vd)]
        if g_pct and not v_pct:   # gold 2.8%, pred 0.028
            alts.append((v * 100, vd - 2))
        if v_pct and not g_pct:   # gold 0.028, pred 2.8%
            alts.append((v / 100, vd + 2))
        if scaled:                # gold in millions, pred in billions (or thousands)
            alts += [(v * 1000, vd - 3), (v / 1000, vd + 3)]
        if abs(v) >= 1e5:         # pred in raw dollars, gold in millions
            alts.append((v / 1e6, vd + 6))
        if any(_round_match(a, ad, g, gd) for a, ad in alts):
            return True
    return False


def is_refusal(pred: str) -> bool:
    return bool(_REFUSAL.search(str(pred)))


# --------------------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are grading an answer to a financial question about a public company filing.
Question: {question}
Gold answer (from a finance expert): {gold}
Gold justification (may be empty): {justification}
Candidate answer: {pred}

Label the candidate with exactly one word:
CORRECT   - same conclusion/number as gold. Allow unit changes (e.g. billions vs millions) and small
            rounding. A different but valid, evidence-consistent assessment also counts as CORRECT.
INCORRECT - wrong number, wrong direction, wrong conclusion, or right number reached with logic that
            contradicts the gold evidence.
REFUSAL   - the candidate says it cannot answer / lacks the information.
Label:"""


def judge(row: dict, pred: str, call_llm) -> str:
    out = call_llm(JUDGE_PROMPT.format(question=row["question"], gold=row["answer"],
                                       justification=row.get("justification") or "", pred=pred))
    w = (out or "").strip().split()[0].upper().strip(".:") if (out or "").strip() else ""
    return {"CORRECT": "correct", "INCORRECT": "incorrect", "REFUSAL": "refusal"}.get(w, "incorrect")


def score_one(row: dict, pred: str, call_llm=None) -> str:
    if row["question_type"] == "metrics-generated":
        if is_refusal(str(pred)[-250:]):   # ends by declining / asking for data
            return "refusal"
        if numeric_match(pred, row["answer"]):
            return "correct"
        return "refusal" if is_refusal(pred) else "incorrect"
    if call_llm is None:
        return "needs_judge"
    return judge(row, pred, call_llm)


# --------------------------------------------------------------------------- reporting
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def score_preds(d_or_rows, preds: list[dict], call_llm=None) -> dict[str, str]:
    """Label every one of the 150 questions. preds: [{"financebench_id", "answer", ("retrieved")}].
    Questions without a prediction get "missing" and count as wrong in report()."""
    rows = {r["financebench_id"]: r for r in (load(d_or_rows) if isinstance(d_or_rows, str) else d_or_rows)}
    ids = [p.get("financebench_id") for p in preds]
    unknown = sorted({i for i in ids if i not in rows}, key=str)
    if unknown:
        raise ValueError(f"{len(unknown)} prediction id(s) not in the dataset, e.g. {unknown[:3]}")
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} id(s) predicted more than once, e.g. {dup[:3]}")
    labels = {i: "missing" for i in rows}
    for p in preds:
        labels[p["financebench_id"]] = score_one(rows[p["financebench_id"]], p.get("answer", ""), call_llm)
    return labels


def report(rows: list[dict], labels: dict[str, str], title: str = "") -> None:
    """Accuracy per question type. Missing predictions count as wrong. A group that still has
    needs_judge items shows counts so far instead of an accuracy."""
    print(f"\n=== {title}")
    miss = sum(1 for r in rows if labels.get(r["financebench_id"], "missing") == "missing")
    if miss:
        print(f"   {miss} of {len(rows)} questions have no prediction: counted as wrong")
    for excl in (False, True):
        groups = defaultdict(list)
        for r in rows:
            if excl and r["financebench_id"] in CONTESTED:
                continue
            lab = labels.get(r["financebench_id"], "missing")
            groups["ALL"].append(lab)
            groups[r["question_type"]].append(lab)
        print(f"-- {'excluding' if excl else 'including'} {len(CONTESTED)} contested items")
        for g, ls in groups.items():
            c = Counter(ls); n = len(ls); k = c["correct"]
            if c["needs_judge"]:     # no accuracy until every item in the group is scored
                print(f"   {g:18s} {c['needs_judge']} of {n} need an LLM judge (pass call_llm); so far "
                      f"correct={k} incorrect={c['incorrect']} refusal={c['refusal']} missing={c['missing']}")
                continue
            lo, hi = wilson(k, n)
            print(f"   {g:18s} n={n:3d}  acc={k/n:6.1%}  95%CI=[{lo:.0%},{hi:.0%}]  "
                  f"incorrect={c['incorrect']}  refusal={c['refusal']}  missing={c['missing']}")


def retrieval_report(rows: list[dict], preds: list[dict], ks=(1, 5, 10, 20)) -> dict:
    """Filing and page recall@k over predictions that include "retrieved" ([doc_name, page0] pairs)."""
    by_id = {r["financebench_id"]: r for r in rows}
    ret = [p for p in preds if p.get("retrieved")]
    out = {}
    if not ret:
        return out
    print(f"   retrieval over {len(ret)} predictions with 'retrieved'")
    for k in ks:
        dr = sum(doc_recall_at_k(p["retrieved"], by_id[p["financebench_id"]]["doc_name"], k) for p in ret) / len(ret)
        pr = sum(page_recall_at_k(p["retrieved"], by_id[p["financebench_id"]]["gold_pages"], k) for p in ret) / len(ret)
        out[k] = (dr, pr)
        print(f"   retrieval@{k:<2d} doc_recall={dr:.2f} page_recall={pr:.2f}")
    return out


def read_preds(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path!r} not found. It is your system's output; to try the scorer first, "
                                f"run `python financebench_starter.py demo` to write one from a released baseline")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def make_demo_preds(d: str, out: str = "preds.jsonl", config: str = "gpt-4-1106-preview_inContext_reverse") -> str:
    """Write a predictions file from one of the 16 released, human-labelled runs (see RESULT_FILES)."""
    with open(f"{d}/results/{config}.jsonl", encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            r = json.loads(line)
            dst.write(json.dumps({"financebench_id": r["financebench_id"], "answer": r["model_answer"]}) + "\n")
    print(f"wrote {out} from {config}")
    return out


# --------------------------------------------------------------------------- judge calibration
def human_labels(d: str) -> list[dict]:
    m = {"Correct Answer": "correct", "Incorrect Answer": "incorrect", "Refusal": "refusal"}
    out = []
    for f in RESULT_FILES:
        with open(f"{d}/results/{f}.jsonl", encoding="utf-8") as fh:
            for r in map(json.loads, fh):
                if r["financebench_id"] not in STALE_HUMAN_LABELS:
                    out.append({**r, "human": m[r["label"]]})
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def calibrate(d: str, scorer, types=None) -> None:
    """scorer(row, model_answer) -> label. Compares against human labels (2,384 usable)."""
    rows = {r["financebench_id"]: r for r in load(d)}
    hs = [h for h in human_labels(d) if types is None or rows[h["financebench_id"]]["question_type"] in types]
    pred = [scorer(rows[h["financebench_id"]], h["model_answer"]) for h in hs]
    gold = [h["human"] for h in hs]
    acc3 = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    accb = sum((p == "correct") == (g == "correct") for p, g in zip(pred, gold)) / len(gold)
    print(f"calibration on {len(gold)} human-labelled completions: 3-way agreement {acc3:.1%}, "
          f"correct-vs-not agreement {accb:.1%}, kappa {cohen_kappa(pred, gold):.2f}")


# --------------------------------------------------------------------------- self-test
def selftest(d: str) -> None:
    rows = load(d)
    assert len(rows) == 150, len(rows)
    print("rows:", len(rows), "| docs referenced:", len({r["doc_name"] for r in rows}),
          "| types:", dict(Counter(r["question_type"] for r in rows)))
    meta = load_meta(d)
    assert meta["FOOTLOCKER_2023_annualreport"]["doc_period"] == 2023
    sizes = {c: len(corpus_docs(d, c)) for c in ("sample", "all", "all+unlabelled")}
    assert sizes == {"sample": 84, "all": 360, "all+unlabelled": 368}, sizes
    files = [f"{d}/data/financebench_open_source.jsonl", f"{d}/data/financebench_document_information.jsonl"]
    assert all(_complete(p, ".jsonl") for p in files)
    print("metadata + corpus modes:", sizes, "| integrity checks: ok")
    met = [r for r in rows if r["question_type"] == "metrics-generated"]
    ok = sum(numeric_match(r["answer"], r["answer"]) for r in met)
    print(f"numeric scorer on gold-as-prediction: {ok}/{len(met)}")
    cases = [("Capex was $1,577 million.", "$1577.00", True), ("Final answer: 0.028", "2.8%", True),
             ("The CCC is (3.7) days", "-3.7", True), ("about $1.58 billion", "$1577.00", True),
             ("Final answer: 24.9", "24.26", False), ("I cannot find this in the filing.", "24.26", False)]
    for p, g, want in cases:
        assert numeric_match(p, g) == want, (p, g)
    assert is_refusal("I cannot find this in the filing.")
    print("unit tests: ok")
    if os.path.isdir(f"{d}/results"):
        calibrate(d, score_one, types={"metrics-generated"})
    pdf_dir = f"{d}/pdfs"
    if os.path.isdir(pdf_dir):
        tok = lambda s: set(re.findall(r"[a-z]{4,}|\d[\d,.]{2,}", s.lower()))
        hit = n = 0
        for r in rows[:40]:
            p = f"{pdf_dir}/{r['doc_name']}.pdf"
            if not os.path.exists(p):
                continue
            pg = pages(p)
            for e in r["evidence"]:
                ev = tok(e["evidence_text_full_page"]); n += 1
                hit += len(ev & tok(pg[e["evidence_page_num"]])) / max(1, len(ev)) > 0.5
        if n:
            print(f"0-indexed page check: {hit}/{n} gold pages match the PDF at evidence_page_num")


def main(argv=None):
    """CLI. In a notebook pass the arguments as a list: main(["score", "--dir", "financebench", "--pred", "preds.jsonl"])."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "selftest", "score", "demo"])
    ap.add_argument("--dir", default="financebench")
    ap.add_argument("--pdfs", action="store_true")
    ap.add_argument("--corpus", default="sample", choices=["sample", "all", "all+unlabelled"])
    ap.add_argument("--pred", help="score: your predictions JSONL; demo: where to write one (default preds.jsonl)")
    ap.add_argument("--config", default="gpt-4-1106-preview_inContext_reverse", choices=RESULT_FILES,
                    help="demo: which released run to turn into a predictions file")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        return setup(a.dir, a.pdfs, a.corpus)
    if a.cmd == "selftest":
        return selftest(a.dir)
    if a.cmd == "demo":
        return make_demo_preds(a.dir, a.pred or "preds.jsonl", a.config)
    if not a.pred:
        ap.error("score needs --pred PATH (your predictions JSONL; `demo` writes an example)")
    rows = load(a.dir)
    preds = read_preds(a.pred)
    labels = score_preds(rows, preds)
    report(rows, labels, title=a.pred)
    retrieval_report(rows, preds)
    return labels


if __name__ == "__main__" and not ("ipykernel" in sys.modules or hasattr(sys, "ps1")):
    main()   # pasted into a notebook cell, the code above only defines the functions: call main([...]) yourself
