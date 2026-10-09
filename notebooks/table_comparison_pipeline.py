"""
Table-aware numerical comparison pipeline for FinQA-style pre-parsed data
(records with a "table" field: a list-of-lists, as in czyssrs/FinQA's
train/dev/test.json -- NOT the BEIR-ified FinanceRAG queries.jsonl, which
strips the table field entirely).

Core idea: keep tables as labeled pandas DataFrames throughout. The LLM's
ONLY job is to identify WHICH row/column labels a comparison question
refers to -- never to do the arithmetic itself. This avoids the ordering-
ambiguity bug in the earlier numeric_line_reasoning.py approach (where
"numbers[1] - numbers[0]" assumed an order that wasn't guaranteed), because
every value is looked up by its real label, not by extraction order.

Pipeline:
    raw FinQA record --parse_finqa_table--> labeled DataFrame
    question + DataFrame --extract_comparison_spec (LLM)--> validated spec
    spec + DataFrame --execute_comparison (pandas + Python)--> numeric answer
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage 1: parse FinQA's raw table field into a labeled DataFrame
# ---------------------------------------------------------------------------

_CELL_NUMBER_RE = re.compile(r"^-?\$?\(?[\d,]+\.?\d*\)?%?$")


def clean_numeric_cell(cell: str) -> float | None:
    """Converts a single table cell to a float if it looks numeric, else
    returns None (so the caller can keep the original string as a label)."""
    cell = cell.strip()
    if not _CELL_NUMBER_RE.match(cell):
        return None
    is_percent = cell.endswith("%")
    cell = cell.rstrip("%")
    paren_negative = cell.startswith("(") and cell.endswith(")")
    if paren_negative:
        cell = cell[1:-1]
    cell = cell.replace(",", "").replace("$", "")
    try:
        value = float(cell)
    except ValueError:
        return None
    if is_percent:
        value /= 100.0
    if paren_negative:
        value = -abs(value)
    return value


def parse_finqa_table(table: list[list[str]]) -> pd.DataFrame:
    """FinQA's table format: table[0] is the header row (first cell often
    blank), table[1:] are data rows with a row label in the first column.
    Returns a DataFrame indexed by row label, columns by header, with
    numeric cells converted to float (non-numeric cells kept as strings,
    e.g. if a row label leaked into a data cell)."""
    header = [h.strip() for h in table[0][1:]]
    row_labels = [row[0].strip() for row in table[1:]]
    data = []
    for row in table[1:]:
        parsed_row = []
        for cell in row[1:]:
            numeric = clean_numeric_cell(cell)
            parsed_row.append(numeric if numeric is not None else cell.strip())
        data.append(parsed_row)
    df = pd.DataFrame(data, index=row_labels, columns=header)
    df.index.name = "line_item"
    return df


# ---------------------------------------------------------------------------
# Stage 2: LLM identifies WHICH labels to compare -- never does the math
# ---------------------------------------------------------------------------

@dataclass
class ComparisonSpec:
    mode: str            # "compare_columns" or "compare_rows"
    row_label: str = None    # used when mode == "compare_columns" (fixed row, two columns)
    col_a: str = None
    col_b: str = None
    row_a: str = None    # used when mode == "compare_rows" (fixed column, two rows)
    row_b: str = None
    col_label: str = None
    operation: str = "difference"   # difference | percent_change | ratio | greater


_SPEC_PROMPT = """You are identifying which cells of a financial table a question refers to.
You will NOT do any arithmetic -- only point at the correct row/column labels.

Table row labels: {row_labels}
Table column labels: {col_labels}

Question: {question}

Respond with ONLY a JSON object, no other text, in one of these two shapes:

If the question compares the SAME line item across two columns (e.g. two years):
{{"mode": "compare_columns", "row_label": "<exact row label>", "col_a": "<exact col label>", "col_b": "<exact col label>", "operation": "<difference|percent_change|ratio|greater>"}}

If the question compares TWO line items within the SAME column:
{{"mode": "compare_rows", "row_a": "<exact row label>", "row_b": "<exact row label>", "col_label": "<exact col label>", "operation": "<difference|percent_change|ratio|greater>"}}

Use "percent_change" for questions about percentage/percent change or growth.
Use "ratio" for "what portion/percentage of" questions.
Use "greater" for yes/no "which was higher/greater" questions.
Otherwise use "difference".
Row/column labels MUST be copied EXACTLY as given above -- do not paraphrase them."""


def make_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "openai":
        import openai
        return openai.OpenAI()
    raise ValueError(f"Unknown provider '{provider}'")


def _call_llm(client, provider: str, model: str, prompt: str) -> str:
    if provider == "anthropic":
        msg = client.messages.create(model=model, max_tokens=200,
                                      messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    elif provider == "openai":
        resp = client.chat.completions.create(model=model, max_tokens=200,
                                               messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    raise ValueError(f"Unknown provider '{provider}'")


def _fuzzy_match_label(target: str, candidates: list[str]) -> str | None:
    """Exact match first; falls back to case-insensitive substring match,
    since LLMs sometimes normalize whitespace/case even when told not to."""
    if target in candidates:
        return target
    target_norm = target.strip().lower()
    for c in candidates:
        if c.strip().lower() == target_norm:
            return c
    for c in candidates:
        if target_norm in c.strip().lower() or c.strip().lower() in target_norm:
            return c
    return None


def extract_comparison_spec(client, provider: str, model: str,
                             question: str, df: pd.DataFrame) -> ComparisonSpec | None:
    """Returns a validated ComparisonSpec, or None if the LLM's response
    can't be parsed or its referenced labels don't actually exist in the
    table -- callers should fall back to free-text reasoning in that case,
    NEVER silently guess a label."""
    prompt = _SPEC_PROMPT.format(
        row_labels=list(df.index), col_labels=list(df.columns), question=question)
    raw = _call_llm(client, provider, model, prompt)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    mode = parsed.get("mode")
    operation = parsed.get("operation", "difference")

    if mode == "compare_columns":
        row_label = _fuzzy_match_label(parsed.get("row_label", ""), list(df.index))
        col_a = _fuzzy_match_label(parsed.get("col_a", ""), list(df.columns))
        col_b = _fuzzy_match_label(parsed.get("col_b", ""), list(df.columns))
        if not (row_label and col_a and col_b):
            return None  # hallucinated/unmatched label -- fail safe, don't guess
        return ComparisonSpec(mode=mode, row_label=row_label, col_a=col_a, col_b=col_b, operation=operation)

    if mode == "compare_rows":
        row_a = _fuzzy_match_label(parsed.get("row_a", ""), list(df.index))
        row_b = _fuzzy_match_label(parsed.get("row_b", ""), list(df.index))
        col_label = _fuzzy_match_label(parsed.get("col_label", ""), list(df.columns))
        if not (row_a and row_b and col_label):
            return None
        return ComparisonSpec(mode=mode, row_a=row_a, row_b=row_b, col_label=col_label, operation=operation)

    return None


# ---------------------------------------------------------------------------
# Stage 3: execute the comparison -- pandas lookup + plain arithmetic
# ---------------------------------------------------------------------------

def _apply_operation(operation: str, value_a: float, value_b: float) -> float | None:
    if operation == "difference":
        return value_b - value_a
    if operation == "percent_change":
        return (value_b - value_a) / value_a if value_a != 0 else None
    if operation == "ratio":
        return value_a / value_b if value_b != 0 else None
    if operation == "greater":
        return 1.0 if value_a > value_b else 0.0
    return None


def execute_comparison(df: pd.DataFrame, spec: ComparisonSpec) -> dict:
    """Looks up the two cells by their REAL labels (no ordering ambiguity)
    and applies the operation. Returns a dict with the numeric result and a
    human-readable explanation for logging/debugging."""
    if spec.mode == "compare_columns":
        value_a = df.loc[spec.row_label, spec.col_a]
        value_b = df.loc[spec.row_label, spec.col_b]
        label_a, label_b = f"{spec.row_label} ({spec.col_a})", f"{spec.row_label} ({spec.col_b})"
    else:  # compare_rows
        value_a = df.loc[spec.row_a, spec.col_label]
        value_b = df.loc[spec.row_b, spec.col_label]
        label_a, label_b = f"{spec.row_a} ({spec.col_label})", f"{spec.row_b} ({spec.col_label})"

    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return {"result": None, "explanation": f"One of the referenced cells is not numeric: "
                                                 f"{label_a}={value_a!r}, {label_b}={value_b!r}"}

    result = _apply_operation(spec.operation, value_a, value_b)
    explanation = f"{spec.operation}({label_a}={value_a}, {label_b}={value_b}) = {result}"
    return {"result": result, "explanation": explanation}


# ---------------------------------------------------------------------------
# End-to-end: one FinQA record + one question -> answer, with fallback
# ---------------------------------------------------------------------------

def answer_comparison_query(client, provider: str, model: str,
                             record: dict, question: str) -> dict:
    """record is one raw FinQA entry (has pre_text/post_text/table/id/qa).
    Returns predicted value, the spec used (or None if fallback fired), and
    an explanation string for debugging/logging."""
    df = parse_finqa_table(record["table"])
    spec = extract_comparison_spec(client, provider, model, question, df)

    if spec is not None:
        outcome = execute_comparison(df, spec)
        if outcome["result"] is not None:
            return {"predicted": outcome["result"], "spec": spec,
                    "explanation": outcome["explanation"], "answer_source": "table_comparison"}

    # Fallback: spec extraction failed, referenced a non-numeric cell, or
    # the question isn't a simple two-cell comparison at all. Reason over
    # the table serialized as text instead of giving up entirely.
    table_text = df.to_string()
    fallback_prompt = (
        f"Using ONLY this table, answer the question with a single number.\n\n"
        f"{table_text}\n\nQuestion: {question}\n\nFinal Answer: <number>"
    )
    raw = _call_llm(client, provider, model, fallback_prompt)
    match = re.search(r"Final Answer:\s*(.+)", raw, re.IGNORECASE)
    predicted = None
    if match:
        num_match = re.search(r"-?[\d,]+\.?\d*", match.group(1))
        if num_match:
            predicted = float(num_match.group().replace(",", ""))
    return {"predicted": predicted, "spec": None, "explanation": raw, "answer_source": "text_fallback"}


# ---------------------------------------------------------------------------
# Evaluation harness against FinQA's own dev/test data
# ---------------------------------------------------------------------------

def answers_match(predicted: float | None, gold: float, rel_tol: float = 0.001) -> bool:
    if predicted is None:
        return False
    return math.isclose(predicted, gold, rel_tol=rel_tol)


_COMPARISON_KEYWORDS = ["difference", "change in", "percent change", "percentage change",
                         "growth", "increase", "decrease", "greater", "higher", "lower",
                         "more than", "less than", "compare", "versus", " vs ", "ratio of",
                         "percentage of", "portion of"]


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _COMPARISON_KEYWORDS)


def evaluate_comparisons(finqa_json_paths: list[str], provider: str, model: str,
                          max_records: int | None = 100) -> pd.DataFrame:
    """Runs the pipeline against FinQA's own dev/test records (which include
    the table field), filtered to comparison-flavored questions only."""
    client = make_client(provider)
    records = []
    for path in finqa_json_paths:
        with open(path) as f:
            records.extend(json.load(f))

    rows = []
    n_checked = 0
    for record in records:
        qa = record.get("qa")
        if not qa or "question" not in qa or "exe_ans" not in qa:
            continue
        if not is_comparison_question(qa["question"]):
            continue
        try:
            gold = float(qa["exe_ans"])
        except (TypeError, ValueError):
            continue

        n_checked += 1
        if max_records and n_checked > max_records:
            break

        outcome = answer_comparison_query(client, provider, model, record, qa["question"])
        rows.append({
            "id": record["id"],
            "question": qa["question"],
            "gold": gold,
            "predicted": outcome["predicted"],
            "answer_source": outcome["answer_source"],
            "spec": outcome["spec"],
            "explanation": outcome["explanation"],
            "correct": answers_match(outcome["predicted"], gold),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparison questions matched.")
        return df

    print(f"Comparison questions evaluated: {len(df)}")
    print(f"Overall accuracy: {df['correct'].mean():.1%}")
    print("\nBy answer source:")
    for source, group in df.groupby("answer_source"):
        print(f"  {source}: {len(group)} -> accuracy {group['correct'].mean():.1%}")
    return df


if __name__ == "__main__":
    df = evaluate_comparisons(
        finqa_json_paths=["./finqa_source/dev.json"],   # add train.json/test.json as needed
        provider="openai",
        model="gpt-4o-mini",
        max_records=100,
    )
    df.to_csv("../results/finqa_comparison_results.csv", index=False)

    wrong = df[~df["correct"]] if not df.empty else df
    if not wrong.empty:
        print("\n=== Sample wrong answers ===")
        for _, r in wrong.head(5).iterrows():
            print(f"\nQ: {r['question']}\ngold: {r['gold']}  predicted: {r['predicted']}  source: {r['answer_source']}")
            print(f"explanation: {r['explanation']}")
