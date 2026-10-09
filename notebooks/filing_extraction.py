"""
Extracts structured tables + surrounding text from raw filings, producing
report dicts in the exact shape earnings_agent.ingest_report() expects:
{"id": ..., "company": ..., "fiscal_period": ..., "pre_text": [...],
 "post_text": [...], "table": [[...]]}.

Primary path: real HTML filings (most SEC EDGAR 10-Ks filed since ~2009
have actual <table> elements) -- pandas.read_html() for parsing +
BeautifulSoup for walking the DOM to grab text immediately before/after
each table. No ML model, no extra service, needs only pandas + lxml/bs4.

Fallback path: anything that isn't clean HTML (scanned PDFs, complex
layouts) -- the `unstructured` library, imported lazily so the primary
path works without it installed at all.
"""

import re
import io

import pandas as pd
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Normalize an arbitrary extracted DataFrame into FinQA's table format
# (list-of-lists: row 0 = header with blank first cell, rows 1+ = row label
# + values) -- this is the shape parse_finqa_table() already expects, so
# nothing downstream needs to change.
# ---------------------------------------------------------------------------

def _flatten_column_label(col) -> str:
    """read_html sometimes returns MultiIndex columns for filings with
    two-level headers (e.g. 'Fiscal Year' / '2017'). Flatten to one string,
    dropping any 'Unnamed: N_level_M' placeholder segments read_html inserts
    for merged/blank header cells."""
    if isinstance(col, tuple):
        parts = [str(p) for p in col if not str(p).startswith("Unnamed:")]
        return " ".join(parts).strip() or str(col[-1])
    return str(col).strip()


def _read_html_safe(io_obj, header):
    """pd.read_html(..., flavor="lxml") raises ImportError (not ValueError)
    if lxml isn't installed -- fall back to letting pandas auto-select
    whatever HTML parser IS available (html5lib, bs4+html.parser) rather
    than hard-failing when lxml is simply missing from this environment."""
    try:
        return pd.read_html(io_obj, header=header, flavor="lxml")
    except ImportError:
        # NOTE: pandas' own flavor=None auto-detection tries lxml FIRST
        # internally, so omitting flavor here would hit the same ImportError
        # again, uncaught -- must name a specific non-lxml flavor instead.
        io_obj.seek(0)   # the stream was partially consumed by the failed attempt
        return pd.read_html(io_obj, header=header, flavor="html5lib")


_YEAR_RE = re.compile(r"^(19|20)\d{2}(\.0)?$")


def clean_edgar_table(raw_df: pd.DataFrame, prior_year_groups: list | None = None):
    """
    Collapses the duplicate-column structure real SEC EDGAR tables are laid
    out with -- confirmed against Apple's actual FY2020 10-K: a year header
    like "2020" repeats across several adjacent raw columns (its XBRL-tagged
    value and its display span land in separate <td>s), and a "$" sign
    claims its own column only on the first data row of a section, shifting
    later rows by one column. A naive header=0 parse leaves every column
    labeled "Unnamed: N" and scatters each real value across 2-3 raw columns.

    Strategy:
      1. Find the header row: the first of the first few rows containing at
         least one 4-digit-year-like token.
      2. Group consecutive columns in that row that share the same year
         label into one logical column (EDGAR repeats the label across its
         sub-columns).
      3. Everything before the first year group is the row-label block --
         collapse it to whichever cell in that span is non-null.
      4. For each data row and each year group, take the first value in that
         column span that parses as a number after stripping "$"/commas --
         this skips the stray "$" cell without hardcoding its exact offset.

    prior_year_groups: the (start, end, label) column groups resolved by the
    MOST RECENT successful call to this function, if any -- pass it in so a
    table with no year header of its own (see the percentage-table case
    above) can inherit column structure instead of failing.

    Returns (DataFrame, year_groups) on success, or (None, None) if this
    table isn't usable -- caller should fall back to the plain header=0
    path for it instead. The returned year_groups should be passed as
    prior_year_groups to the NEXT call, continuing the chain.
    """
    def cell_str(v) -> str:
        """Safe string conversion -- .astype(str) doesn't reliably turn NaN
        into "nan" for every dtype/pandas-version combination, confirmed
        against this real filing's parsed tables, so every cell read in
        this function goes through this instead of trusting astype(str)."""
        return "" if pd.isna(v) else str(v).strip()

    def normalize_year(v: str) -> str:
        """Collapses "2020" / "2020.0" to the same label -- EDGAR's raw
        parsed header repeats a year across 2-3 sub-columns, but pandas
        represents the SAME year differently in different sub-columns
        (int-like string vs float string) depending on that column's
        inferred dtype, so an exact-string grouping missed the duplication
        entirely until this normalization was added."""
        return v[:-2] if _YEAR_RE.match(v) and v.endswith(".0") else v

    def first_numeric_in_span(row, start, end):
        for c in range(start, end + 1):
            val = cell_str(row.iloc[c])
            if val in ("", "$"):
                continue
            cleaned = val.replace(",", "").replace("$", "")
            try:
                float(cleaned)
                return val
            except ValueError:
                continue
        return ""   # no numeric value found in this span -- leave blank, not a guess

    def first_nonnull_in_span(row, start, end):
        for c in range(start, end + 1):
            val = cell_str(row.iloc[c])
            if val:
                return val
        return ""

    def build_result(header_row_idx, label_groups, year_groups):
        ncols = raw_df.shape[1]
        # Borrowed column boundaries (from a prior table's header-inheritance
        # case) can exceed THIS table's actual width -- different raw tables
        # representing "the same" N-year layout don't always parse to the
        # same column count, confirmed against real filing data. Clamp
        # rather than crash, and bail out if clamping leaves nothing usable.
        label_groups = [(s, min(e, ncols - 1), lbl) for s, e, lbl in label_groups if s < ncols]
        year_groups = [(s, min(e, ncols - 1), label) for s, e, label in year_groups if s < ncols]
        if not label_groups or not year_groups:
            return None

        out_rows = []
        for r in range(header_row_idx + 1, len(raw_df)):
            row = raw_df.iloc[r]
            row_label = first_nonnull_in_span(row, label_groups[0][0], label_groups[-1][1])
            if not row_label:
                continue   # a fully-blank row (pure spacer row) -- skip it
            values = [first_numeric_in_span(row, start, end) for start, end, _ in year_groups]
            out_rows.append([row_label] + values)
        if not out_rows:
            return None
        columns = [""] + [label for _, _, label in year_groups]
        return pd.DataFrame(out_rows, columns=columns)

    header_row_idx = None
    for i in range(min(3, len(raw_df))):
        row_vals = [cell_str(v) for v in raw_df.iloc[i]]
        if any(_YEAR_RE.match(v) for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        if prior_year_groups is None:
            return None, None
        # Real EDGAR filings often split one logical table into a dollar-
        # amount table and a percentage table right after it, with the
        # percentage table NOT repeating the year header at all --
        # confirmed against Apple's actual FY2020 10-K ("Gross margin:"
        # followed immediately by "Gross margin percentage:"). Reuse the
        # previous table's column boundaries rather than failing outright.
        label_groups = [(0, prior_year_groups[0][0] - 1, "")]
        result = build_result(header_row_idx=-1, label_groups=label_groups, year_groups=prior_year_groups)
        return result, (prior_year_groups if result is not None else None)

    header_row = pd.Series([normalize_year(cell_str(v)) for v in raw_df.iloc[header_row_idx]])

    # Group consecutive columns by header value, recording each group's
    # (start_col, end_col, label).
    groups = []
    col = 0
    ncols = raw_df.shape[1]
    while col < ncols:
        label = header_row.iloc[col]
        start = col
        while col + 1 < ncols and header_row.iloc[col + 1] == label:
            col += 1
        groups.append((start, col, label))
        col += 1

    # Everything before the first genuinely year-like group is the row-label
    # block; everything from the first year group onward is data.
    year_group_indices = [i for i, (_, _, label) in enumerate(groups) if _YEAR_RE.match(label)]
    if not year_group_indices:
        return None, None
    label_groups = groups[: year_group_indices[0]]
    year_groups = [groups[i] for i in year_group_indices]

    result = build_result(header_row_idx, label_groups, year_groups)
    return result, (year_groups if result is not None else None)


def dataframe_to_finqa_table(df: pd.DataFrame) -> list[list[str]] | None:
    """Returns None if this DataFrame doesn't look like a usable financial
    table (too small, or no plausible row-label column) rather than forcing
    a bad conversion -- callers should skip these, not pass them through."""
    if df.shape[0] < 1 or df.shape[1] < 2:
        return None  # need at least one data row and a label col + one value col

    df = df.copy()
    df.columns = [_flatten_column_label(c) for c in df.columns]

    # Heuristic: the first column is the row-label column if it's mostly
    # non-numeric text (line-item names) rather than numbers -- if it looks
    # numeric instead, this table probably has no natural row labels and
    # isn't a good fit for label-based operations downstream.
    first_col = df.iloc[:, 0].astype(str)
    numeric_like = first_col.str.match(r"^-?\$?\(?[\d,]+\.?\d*\)?%?$").mean()
    if numeric_like > 0.5:
        return None

    header_row = [""] + list(df.columns[1:])
    data_rows = [[str(row[0])] + [str(v) for v in row[1:]] for row in df.itertuples(index=False)]
    return [header_row] + data_rows


# ---------------------------------------------------------------------------
# Primary path: real HTML, via BeautifulSoup (context) + read_html (parsing)
# ---------------------------------------------------------------------------

def _nearby_text(tag, direction: str, max_blocks: int = 2) -> list[str]:
    """Walks siblings before ('prev') or after ('next') a <table> tag,
    collecting up to max_blocks non-empty text blocks -- this is the
    pre_text/post_text context a table needs to be interpretable on its own
    (e.g. 'the following table summarizes...' immediately before it)."""
    blocks = []
    sibling_iter = tag.find_previous_siblings() if direction == "prev" else tag.find_next_siblings()
    for sib in sibling_iter:
        text = sib.get_text(" ", strip=True)
        if text:
            blocks.append(text)
        if len(blocks) >= max_blocks:
            break
    if direction == "prev":
        blocks.reverse()   # find_previous_siblings walks backward; restore reading order
    return blocks


def extract_tables_from_html(html: str, company: str = None, fiscal_period: str = None,
                              id_prefix: str = "filing") -> list[dict]:
    """Main entry point for HTML filings. Returns a list of report dicts,
    one per usable table found -- tables that fail dataframe_to_finqa_table's
    sanity check are skipped, not force-included as garbage."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # lxml not installed in this environment -- fall back to Python's
        # built-in parser rather than hard-failing the whole batch. Slower
        # and marginally less lenient on malformed markup, but functional.
        import warnings
        warnings.warn(
            "lxml not available, falling back to the built-in html.parser. "
            "Run 'pip install lxml' for faster, more robust parsing of large filings.",
            stacklevel=2,
        )
        soup = BeautifulSoup(html, "html.parser")
    table_tags = soup.find_all("table")
    reports = []
    prior_year_groups = None   # carried across tables for the header-inheritance case

    for i, tag in enumerate(table_tags):
        try:
            raw_dfs = _read_html_safe(io.StringIO(str(tag)), header=None)
        except ValueError:
            continue   # pandas couldn't parse this <table> at all -- skip, don't crash the whole filing
        if not raw_dfs:
            continue
        raw_df = raw_dfs[0]

        # Try the EDGAR duplicate-column cleanup first (handles real
        # multi-period financial tables, including ones that inherit their
        # column structure from the table right before them); fall back to
        # a plain header=0 read for simpler tables (e.g. the filing's
        # cover-page tables) that don't have this structure at all.
        cleaned, resolved_year_groups = clean_edgar_table(raw_df, prior_year_groups)
        if cleaned is not None:
            finqa_table = dataframe_to_finqa_table(cleaned)
            prior_year_groups = resolved_year_groups
        else:
            try:
                dfs = _read_html_safe(io.StringIO(str(tag)), header=0)
                finqa_table = dataframe_to_finqa_table(dfs[0]) if dfs else None
            except ValueError:
                finqa_table = None
        if finqa_table is None:
            continue

        reports.append({
            "id": f"{id_prefix}_table_{i}",
            "company": company,
            "fiscal_period": fiscal_period,
            "pre_text": _nearby_text(tag, "prev"),
            "post_text": _nearby_text(tag, "next"),
            "table": finqa_table,
        })

    return reports


# ---------------------------------------------------------------------------
# Fallback path: unstructured library, for non-HTML or messy documents.
# Imported lazily so the primary path above works without this dependency.
# ---------------------------------------------------------------------------

def extract_tables_via_unstructured(file_path: str, company: str = None,
                                     fiscal_period: str = None, id_prefix: str = "filing") -> list[dict]:
    """Use when extract_tables_from_html finds nothing usable, or the source
    isn't HTML at all (a scanned or complex-layout PDF). Requires:
    pip install unstructured[pdf] --break-system-packages
    (plus system deps like poppler-utils for the PDF path)."""
    from unstructured.partition.auto import partition

    elements = partition(filename=file_path, strategy="hi_res")
    reports = []
    table_index = 0

    for i, element in enumerate(elements):
        if element.category != "Table":
            continue

        html_repr = getattr(element.metadata, "text_as_html", None)
        if not html_repr:
            continue   # this element was detected as a table but has no parseable structure
        try:
            dfs = pd.read_html(io.StringIO(html_repr), header=0)
        except ValueError:
            continue
        if not dfs:
            continue

        finqa_table = dataframe_to_finqa_table(dfs[0])
        if finqa_table is None:
            continue

        # Grab a couple of narrative elements immediately before/after this
        # table in the document's reading order for pre_text/post_text.
        pre_text = [e.text for e in elements[max(0, i - 2):i] if e.category == "NarrativeText" and e.text.strip()]
        post_text = [e.text for e in elements[i + 1:i + 3] if e.category == "NarrativeText" and e.text.strip()]

        reports.append({
            "id": f"{id_prefix}_table_{table_index}",
            "company": company,
            "fiscal_period": fiscal_period,
            "pre_text": pre_text,
            "post_text": post_text,
            "table": finqa_table,
        })
        table_index += 1

    return reports


# ---------------------------------------------------------------------------
# Combined entry point: try HTML first, fall back to unstructured if HTML
# parsing found nothing usable at all.
# ---------------------------------------------------------------------------

def extract_reports_from_filing(source: str, company: str = None, fiscal_period: str = None,
                                 id_prefix: str = "filing", is_html: bool = True) -> list[dict]:
    """source: HTML string (is_html=True) or a file path (is_html=False,
    routes straight to the unstructured fallback -- for PDFs/scanned docs)."""
    if is_html:
        reports = extract_tables_from_html(source, company, fiscal_period, id_prefix)
        if reports:
            return reports
        print("No usable tables found via HTML parsing -- falling back to unstructured "
              "(requires the source to be saved to a file for this path).")
        return []
    return extract_tables_via_unstructured(source, company, fiscal_period, id_prefix)


# ---------------------------------------------------------------------------
# Markdown-table path: for filings pulled through any fetcher that converts
# HTML to markdown before you see it (common in scraping/RAG tooling) --
# verified against real content from Apple's actual FY2023 10-K, since a
# live fetch of the raw HTML wasn't available to test the path above against.
# ---------------------------------------------------------------------------

def _parse_markdown_table_block(lines: list[str]) -> list[list[str]]:
    """lines: consecutive '| ... |' lines (including the '| --- |' separator
    row, which is dropped). Returns a list-of-lists of cell strings."""
    rows = []
    for line in lines:
        if re.fullmatch(r"\|[\s:|-]+\|", line.strip()):
            continue   # the '| --- | --- |' separator row -- not data
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def extract_tables_from_markdown(markdown_text: str, company: str = None, fiscal_period: str = None,
                                  id_prefix: str = "filing") -> list[dict]:
    """Splits markdown text into contiguous '|'-line blocks (each is one
    table) and non-table text blocks (pre_text/post_text context), same
    role as extract_tables_from_html but for pre-converted markdown input."""
    lines = markdown_text.split("\n")
    blocks = []   # list of {"type": "table"|"text", "lines": [...]}
    current_type = None
    current_lines = []

    for line in lines:
        is_table_line = line.strip().startswith("|")
        line_type = "table" if is_table_line else "text"
        if line_type != current_type and current_lines:
            blocks.append({"type": current_type, "lines": current_lines})
            current_lines = []
        current_type = line_type
        current_lines.append(line)
    if current_lines:
        blocks.append({"type": current_type, "lines": current_lines})

    reports = []
    table_index = 0
    last_good_header = None
    for i, block in enumerate(blocks):
        if block["type"] != "table":
            continue

        raw_rows = _parse_markdown_table_block([l for l in block["lines"] if l.strip()])
        if len(raw_rows) < 2:
            continue   # need at least a header + one data row

        header, data = raw_rows[0], raw_rows[1:]

        # Real SEC filings sometimes split one logical table into consecutive
        # blocks that SHARE column headers only stated once (e.g. a "gross
        # margin" dollar table immediately followed by a "gross margin
        # percentage" table with no repeated year row) -- confirmed against
        # Apple's actual FY2023 10-K. Detect a header that's blank except
        # possibly its first cell, and inherit the previous table's header
        # instead of silently producing unusable blank column labels.
        if all(not h.strip() for h in header[1:]) and last_good_header is not None:
            print(f"Warning: table block {i} has a blank header row -- inheriting column "
                  f"labels {last_good_header[1:]} from the preceding table. Verify this is "
                  f"correct for your document; it's a heuristic, not a guarantee.")
            header = last_good_header
        else:
            last_good_header = header

        try:
            df = pd.DataFrame(data, columns=header if len(header) == len(data[0]) else None)
        except ValueError:
            continue   # ragged rows (inconsistent column counts) -- skip rather than force it
        finqa_table = dataframe_to_finqa_table(df)
        if finqa_table is None:
            continue

        pre_text = [l.strip() for l in blocks[i - 1]["lines"] if l.strip()][-2:] if i > 0 else []
        post_text = [l.strip() for l in blocks[i + 1]["lines"] if l.strip()][:2] if i + 1 < len(blocks) else []

        reports.append({
            "id": f"{id_prefix}_table_{table_index}",
            "company": company, "fiscal_period": fiscal_period,
            "pre_text": pre_text, "post_text": post_text, "table": finqa_table,
        })
        table_index += 1

    return reports
