"""
Walks a local directory of downloaded SEC filings structured as:
    <root>/<TICKER>/<FORM_TYPE>/<DATE>/<ACCESSION_NUMBER>/*.htm

e.g. D:\\Fin_Rag\\sec_filings\\AAPL\\10-K\\2020-09-26\\0000320193-20-000096\\

For each accession folder, identifies the main filing document (not an
exhibit/certification file), reads its RAW HTML from disk, and runs it
through extract_reports_from_filing() -- this is genuine raw HTML, unlike
anything obtainable through a markdown-converting web fetch, so this is
the first real test of that code path.

Usage:
    python batch_ingest_filings.py "D:\\Fin_Rag\\sec_filings" --write-jsonl reports.jsonl
    # or, to ingest straight into earnings_agent's Postgres + Chroma stores:
    python batch_ingest_filings.py "D:\\Fin_Rag\\sec_filings" --ingest
"""

import argparse
import json
from pathlib import Path

from filing_extraction import extract_reports_from_filing

# Filenames matching these patterns are exhibits/certifications/graphics,
# not the main filing body -- skip them when picking the document to parse.
_EXCLUDE_PATTERNS = ["ex-", "ex_", "-ex", "_ex", "cert", "graphic", "image", "logo"]


def find_main_document(accession_dir: Path, ticker: str, date_str: str) -> Path | None:
    """Picks the primary filing document out of an accession folder, which
    typically also contains exhibits, XBRL instance files, and images.

    Strategy, in order:
      1. Exact match on SEC's usual naming convention: {ticker}-{yyyymmdd}.htm
      2. Any .htm/.html file NOT matching an exhibit-like pattern, preferring
         the largest by file size (the main document is almost always the
         biggest file in the folder -- exhibits are short).
    """
    htm_files = list(accession_dir.glob("*.htm")) + list(accession_dir.glob("*.html"))
    if not htm_files:
        return None

    compact_date = date_str.replace("-", "")
    expected_name = f"{ticker.lower()}-{compact_date}.htm"
    for f in htm_files:
        if f.name.lower() == expected_name:
            return f

    candidates = [f for f in htm_files if not any(p in f.name.lower() for p in _EXCLUDE_PATTERNS)]
    if not candidates:
        candidates = htm_files   # fall back to everything if the exclude filter removed all of them

    return max(candidates, key=lambda f: f.stat().st_size)


def parse_filing_path(accession_dir: Path, root: Path) -> dict:
    """Extracts ticker/form_type/date/accession from the folder structure
    itself -- this metadata is already encoded in the path, so no need to
    parse it out of the filing content."""
    rel_parts = accession_dir.relative_to(root).parts
    if len(rel_parts) < 4:
        raise ValueError(f"Unexpected path depth under root: {accession_dir}")
    ticker, form_type, date_str, accession = rel_parts[-4:]
    return {"ticker": ticker, "form_type": form_type, "date": date_str, "accession": accession}


def find_all_accession_dirs(root: Path) -> list[Path]:
    """An accession dir is any directory at exactly ticker/form/date/accession
    depth under root that contains at least one .htm/.html file."""
    accession_dirs = []
    for candidate in root.glob("*/*/*/*"):
        if candidate.is_dir() and (list(candidate.glob("*.htm")) or list(candidate.glob("*.html"))):
            accession_dirs.append(candidate)
    return accession_dirs


def read_html_file(path: Path) -> str:
    """SEC filings are usually UTF-8, but older ones can be latin-1 --
    try UTF-8 first, fall back rather than crashing the whole batch on one
    bad file."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  UTF-8 decode failed for {path.name}, retrying as latin-1")
        return path.read_text(encoding="latin-1")


def process_filing_directory(root_dir: str) -> list[dict]:
    root = Path(root_dir)
    accession_dirs = find_all_accession_dirs(root)
    print(f"Found {len(accession_dirs)} filing folders under {root}")

    all_reports = []
    for accession_dir in accession_dirs:
        try:
            meta = parse_filing_path(accession_dir, root)
        except ValueError as e:
            print(f"  Skipping {accession_dir}: {e}")
            continue

        main_doc = find_main_document(accession_dir, meta["ticker"], meta["date"])
        if main_doc is None:
            print(f"  No .htm/.html file found in {accession_dir}, skipping")
            continue

        print(f"\n{meta['ticker']} {meta['form_type']} {meta['date']} -> {main_doc.name}")
        try:
            html = read_html_file(main_doc)
        except Exception as e:
            print(f"  Failed to read {main_doc}: {e}")
            continue

        id_prefix = f"{meta['ticker']}_{meta['form_type']}_{meta['date']}_{meta['accession']}"
        try:
            reports = extract_reports_from_filing(
                html, company=meta["ticker"], fiscal_period=meta["date"],
                id_prefix=id_prefix, is_html=True,
            )
        except Exception as e:
            # A parsing failure on ONE filing shouldn't kill the whole batch --
            # log it and keep going, same philosophy as the per-table skip
            # logic inside extract_tables_from_html itself.
            print(f"  extract_reports_from_filing failed for {main_doc}: {e}")
            continue

        print(f"  Extracted {len(reports)} table(s)")
        all_reports.extend(reports)

    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help=r"Root of the downloaded filings, e.g. D:\Fin_Rag\sec_filings")
    parser.add_argument("--write-jsonl", help="Path to write extracted reports as JSON Lines")
    parser.add_argument("--ingest", action="store_true",
                         help="Ingest extracted reports directly into earnings_agent's Postgres + Chroma stores")
    args = parser.parse_args()

    all_reports = process_filing_directory(args.root_dir)
    print(f"\nTotal tables extracted across all filings: {len(all_reports)}")

    if args.write_jsonl:
        with open(args.write_jsonl, "w") as f:
            for r in all_reports:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(all_reports)} reports to {args.write_jsonl}")

    if args.ingest:
        from earnings_agent import ingest_report, init_postgres_schema
        init_postgres_schema()
        for r in all_reports:
            ingest_report(r)
        print(f"Ingested {len(all_reports)} reports into Postgres + Chroma")
