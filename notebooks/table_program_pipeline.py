"""
Generalized table-query pipeline: replaces the fixed-mode ComparisonSpec
approach with a small, composable operation vocabulary + a step-by-step
program executor. This is deliberately the same architectural shape as
czyssrs/FinQA's own DSL (subtract/divide/table_max/etc. + executor) --
we're not training a model to generate programs, we're having the LLM
COMPOSE one out of a fixed, safe vocabulary, validated against real table
labels at every step.

Why this generalizes where fixed modes didn't: "lookup", "difference",
"percent_change", "trend over N years", "ratio of total", "which period was
highest" are ALL just short programs built from the same ~9 primitives
below. Adding support for a genuinely new question shape means writing a
new PROMPT EXAMPLE, not new Python code -- the executor doesn't change.

Safety note: this is NOT free-form code execution (unlike e.g. LangChain's
pandas dataframe agent, which runs LLM-generated Python and carries a real
code-injection risk). The LLM can only reference a fixed, small set of named
operations and real table cells -- there is no way for it to execute
arbitrary code through this interface.
"""

import json
import re
from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Operation vocabulary -- the ONLY things a program can do. Adding a new
# primitive (e.g. "median") is a one-line addition here; it does not require
# touching the executor or the spec-validation logic.
# ---------------------------------------------------------------------------

def _op_add(args): return args[0] + args[1]
def _op_subtract(args): return args[0] - args[1]          # subtract(a, b) = a - b
def _op_multiply(args): return args[0] * args[1]
def _op_divide(args): return args[0] / args[1] if args[1] != 0 else None
def _op_average(args): return sum(args) / len(args) if args else None
def _op_max(args): return max(args) if args else None
def _op_min(args): return min(args) if args else None
def _op_greater(args): return 1.0 if args[0] > args[1] else 0.0
def _op_sum(args): return sum(args)

OPERATIONS = {
    "add": _op_add, "subtract": _op_subtract, "multiply": _op_multiply,
    "divide": _op_divide, "average": _op_average, "max": _op_max,
    "min": _op_min, "greater": _op_greater, "sum": _op_sum,
}


# ---------------------------------------------------------------------------
# Program representation + validation
# ---------------------------------------------------------------------------

@dataclass
class ProgramStep:
    op: str
    args: list   # each arg is: a literal number, {"row": ..., "col": ...}, or "#N" (prior step's result)


@dataclass
class ExecutionResult:
    final_value: float | None
    trace: list[str] = field(default_factory=list)   # human-readable log of every step, for debugging
    error: str | None = None


def _fuzzy_match_label(target: str, candidates: list[str]) -> str | None:
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


def _resolve_arg(arg, df: pd.DataFrame, step_results: list[float]):
    """Turns one program argument into a concrete number OR a list of
    numbers. Four valid shapes, checked in order:
      - a plain number: used as-is
      - {"row": ..., "col": ...}: a single table cell, validated against
        REAL labels
      - {"row": ..., "col": "all"}: EVERY numeric cell in that row, in
        column order -- lets a program stay short regardless of how many
        periods the table has (10 years is the same one-line arg as 3)
      - "#N": the result of an earlier step in this same program (may
        itself be a list, e.g. from a "col": "all" resolution, if that
        step's op accepts list-valued args like average/max/min/sum)
    Anything else raises -- this is the safety boundary: no other shape of
    argument is ever accepted, so there's no way to smuggle arbitrary
    behavior through here."""
    if isinstance(arg, (int, float)):
        return float(arg)

    if isinstance(arg, str) and re.fullmatch(r"#\d+", arg):
        idx = int(arg[1:])
        if idx >= len(step_results):
            raise ValueError(f"Reference {arg} points to a step that hasn't run yet")
        return step_results[idx]

    if isinstance(arg, dict) and "row" in arg and "col" in arg:
        row = _fuzzy_match_label(arg["row"], list(df.index))
        if row is None:
            raise ValueError(f"Cell reference {arg} does not match any real row label")

        if arg["col"] == "all":
            values = [v for v in df.loc[row] if isinstance(v, (int, float))]
            if not values:
                raise ValueError(f"Row '{row}' has no numeric columns to expand with 'all'")
            return [float(v) for v in values]

        col = _fuzzy_match_label(arg["col"], list(df.columns))
        if col is None:
            raise ValueError(f"Cell reference {arg} does not match any real column label")
        value = df.loc[row, col]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Cell ({row}, {col}) is not numeric: {value!r}")
        return float(value)

    raise ValueError(f"Malformed argument: {arg!r}")


def execute_program(df: pd.DataFrame, program: list[dict]) -> ExecutionResult:
    """Runs a program step by step. Fails safely on the first invalid step
    (bad op name, hallucinated label, malformed arg) rather than guessing --
    trace shows exactly how far it got before failing, which is the debugging
    payoff of this whole design over free-form LLM arithmetic."""
    step_results: list[float] = []
    trace: list[str] = []

    for i, step in enumerate(program):
        op_name = step.get("op")
        raw_args = step.get("args", [])

        if op_name not in OPERATIONS:
            return ExecutionResult(None, trace, error=f"Step {i}: unknown operation '{op_name}'")

        try:
            resolved_args = [_resolve_arg(a, df, step_results) for a in raw_args]
            # "col": "all" resolves to a LIST. If it's the sole argument,
            # unwrap it as the full args list (so average/max/min/sum can
            # take a single row-expansion arg instead of one arg per period).
            # A list-valued arg alongside other args is ambiguous for
            # binary ops (add/subtract/etc. expect exactly two scalars) and
            # is rejected explicitly rather than silently guessing.
            if len(resolved_args) == 1 and isinstance(resolved_args[0], list):
                flat_args = resolved_args[0]
            elif any(isinstance(a, list) for a in resolved_args):
                raise ValueError(
                    "A 'col': 'all' argument can only be used as the SOLE "
                    "argument to average/max/min/sum, not mixed with other args"
                )
            else:
                flat_args = resolved_args
            result = OPERATIONS[op_name](flat_args)
        except (ValueError, ZeroDivisionError, KeyError) as e:
            return ExecutionResult(None, trace, error=f"Step {i} ('{op_name}'): {e}")

        step_results.append(result)
        trace.append(f"#{i}: {op_name}({raw_args}) = {result}")

    final_value = step_results[-1] if step_results else None
    return ExecutionResult(final_value, trace)


# ---------------------------------------------------------------------------
# LLM composes the program -- never executes it, never does arithmetic itself
# ---------------------------------------------------------------------------

_PROGRAM_PROMPT = """You are composing a short PROGRAM to answer a question about a financial
table, using ONLY the operations listed below. You do NOT compute the
answer yourself -- a separate executor will run your program.

Table row labels: {row_labels}
Table column labels: {col_labels}

Available operations (each takes a list of args):
  add(a, b), subtract(a, b) = a - b, multiply(a, b), divide(a, b),
  average(list), max(list), min(list), greater(a, b) -> 1 if a>b else 0, sum(list)

Each argument to an operation must be ONE of:
  - a literal number
  - {{"row": "<exact row label>", "col": "<exact col label>"}} -- a single table cell
  - {{"row": "<exact row label>", "col": "all"}} -- EVERY numeric value in that
    row across all columns, as a list. Use this for average/max/min/sum over
    every available period instead of naming each column separately -- it
    works the same whether the table has 3 columns or 30.
  - "#N" -- the result of step N (0-indexed) earlier in YOUR OWN program

Respond with ONLY a JSON array of steps, no other text. Example shapes:

Simple fact lookup ("what was revenue in 2017?"):
[{{"op": "add", "args": [{{"row": "total revenue", "col": "2017"}}, 0]}}]

Two-point difference ("increase in net income from 2016 to 2017?"):
[{{"op": "subtract", "args": [{{"row": "net income", "col": "2017"}}, {{"row": "net income", "col": "2016"}}]}}]

Percent change (chains TWO steps, second step references the first via "#0"):
[{{"op": "subtract", "args": [{{"row": "net income", "col": "2017"}}, {{"row": "net income", "col": "2016"}}]}},
 {{"op": "divide", "args": ["#0", {{"row": "net income", "col": "2016"}}]}}]

Trend/growth over ALL available years for one line item ("is net income growing?"):
list every available year's cell for that row, then subtract earliest from latest,
then divide by the earliest -- e.g. for 3 years 2015/2016/2017:
[{{"op": "subtract", "args": [{{"row": "net income", "col": "2017"}}, {{"row": "net income", "col": "2015"}}]}},
 {{"op": "divide", "args": ["#0", {{"row": "net income", "col": "2015"}}]}}]

Which period was highest, regardless of how many columns exist -- use "all"
rather than listing every column by name:
[{{"op": "max", "args": [{{"row": "total revenue", "col": "all"}}]}}]

Average across every available period for a row, regardless of column count:
[{{"op": "average", "args": [{{"row": "operating margin", "col": "all"}}]}}]

Row/column labels in your program MUST be copied EXACTLY from the lists above.
Question: {question}"""


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
        msg = client.messages.create(model=model, max_tokens=400,
                                      messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    elif provider == "openai":
        resp = client.chat.completions.create(model=model, max_tokens=400,
                                               messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    raise ValueError(f"Unknown provider '{provider}'")


def extract_program(client, provider: str, model: str, question: str, df: pd.DataFrame) -> list[dict] | None:
    prompt = _PROGRAM_PROMPT.format(row_labels=list(df.index), col_labels=list(df.columns), question=question)
    raw = _call_llm(client, provider, model, prompt)
    try:
        program = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(program, list) or not program:
        return None
    return program


_RETRY_PROMPT = """Your previous program failed to execute. Fix it and respond with ONLY
the corrected JSON array, no other text.

Table row labels: {row_labels}
Table column labels: {col_labels}
Question: {question}

Your previous program:
{previous_program}

Error: {error}

Common causes: a row/column label that doesn't exactly match one of the
labels listed above, or mixing a "col": "all" argument with other arguments
in the same operation. Correct the program accordingly."""


def extract_program_with_retry(client, provider: str, model: str, question: str, df: pd.DataFrame,
                                max_retries: int = 2) -> tuple[list[dict] | None, ExecutionResult | None]:
    """Tries extract_program, executes it, and on failure feeds the actual
    execution error back to the LLM for a corrected attempt -- up to
    max_retries times. This meaningfully reduces the "hallucinated label"
    and "malformed args" failure modes without any change to the executor
    itself, since the LLM gets concrete, specific feedback rather than
    guessing blind on a second try."""
    program = extract_program(client, provider, model, question, df)
    if program is None:
        return None, None

    outcome = execute_program(df, program)
    attempts = 0
    while outcome.error is not None and attempts < max_retries:
        attempts += 1
        retry_prompt = _RETRY_PROMPT.format(
            row_labels=list(df.index), col_labels=list(df.columns), question=question,
            previous_program=json.dumps(program), error=outcome.error,
        )
        raw = _call_llm(client, provider, model, retry_prompt)
        try:
            program = json.loads(raw)
        except json.JSONDecodeError:
            continue  # try again next loop iteration if attempts remain, else exit with last error
        if not isinstance(program, list) or not program:
            continue
        outcome = execute_program(df, program)

    return program, outcome


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def answer_query(client, provider: str, model: str, df: pd.DataFrame, question: str,
                  max_retries: int = 2) -> dict:
    program, outcome = extract_program_with_retry(client, provider, model, question, df, max_retries)

    if program is None:
        return {"predicted": None, "program": None, "trace": [],
                "error": "LLM did not return a parseable program (even after retries)",
                "answer_source": "program_failed"}

    if outcome.error is not None:
        # Retries exhausted without a working program -- fail safe with the
        # full trace rather than guessing. Text-fallback (reasoning over the
        # serialized table, same pattern as finqa_reasoning_eval.py) is a
        # natural next addition here if this still fires too often in practice.
        return {"predicted": None, "program": program, "trace": outcome.trace,
                "error": outcome.error, "answer_source": "program_failed"}

    return {"predicted": outcome.final_value, "program": program, "trace": outcome.trace,
            "error": None, "answer_source": "program_executed"}
