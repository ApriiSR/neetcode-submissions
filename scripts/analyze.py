#!/usr/bin/env python3
"""Analyze NeetCode submissions: golf stats + LLM complexity analysis.

Scans `Data Structures & Algorithms/<slug>/submission-N.py`, writes
`analysis/<slug>/submission-N.json` for any submission missing one, then
regenerates `analysis/summary.json`.

Usage:
    python3 scripts/analyze.py [--mock] [--only SLUG]
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generators
import statements

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_ROOT = REPO_ROOT / "Data Structures & Algorithms"
ANALYSIS_ROOT = REPO_ROOT / "analysis"
BENCHMARKS_ROOT = ANALYSIS_ROOT / "benchmarks"

# Where this run's list of newly-analyzed submissions gets written, for
# scripts/announce.py to pick up — a scratch file, never committed (see
# .gitignore). Overridable via env for callers that want a different
# location; defaults under ANALYSIS_ROOT so test monkeypatching of that
# still isolates it per-test.
NEW_THIS_RUN_ENV = "ANALYZE_NEW_THIS_RUN_PATH"

# `or` fallbacks: the workflow exports these from repo vars, which arrive as
# empty strings when unset.
# Defaults target Kimi-for-Coding (key from kimi.com/code/console, billed to the
# subscription). That endpoint requires model id exactly "k3" and rejects any
# temperature other than 1. The open-platform equivalent is
# https://api.moonshot.ai/v1 with model "kimi-k3".
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL") or "https://api.kimi.com/coding/v1"
MOONSHOT_MODEL = os.environ.get("MOONSHOT_MODEL") or "k3"
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")

SUBMISSION_RE = re.compile(r"^submission-(\d+)\.py$")

# Analysis records below this version are re-analyzed rather than skipped —
# see has_good_analysis(). Bumped to 2 when the problem statement was added
# to the prompt, then to 3 when the statement source switched from
# LeetCode (via a slug mapping) to NeetCode's own endpoint — NeetCode's
# stated constraints can differ from LeetCode's (e.g. buy-and-sell-crypto
# caps prices at 100 on NeetCode vs. 10^4 on LeetCode), so every record
# analyzed against a LeetCode-sourced statement needs redoing against the
# correct one.
ANALYSIS_VERSION = 3

# Defensive cap on how much of a fetched problem statement goes into the
# prompt. NeetCode statements observed so far top out well under this; it
# exists to bound prompt size/cost if a future fetch returns something huge.
STATEMENT_MAX_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a precise algorithms reviewer. Given a problem slug, optionally "
    "its problem statement, a note on how that problem's input scales with "
    "n, and a Python solution, respond with STRICT JSON only (no markdown "
    "fences, no prose outside the JSON object) with exactly these fields: "
    '"time_average" (string, Big-O), "time_worst" (string, Big-O; if the '
    "solution uses a dict or set, the worst case must account for adversarial "
    "hash collisions degrading lookups to O(n)), \"space\" (string, Big-O), "
    '"hash_dependent" (boolean, true if correctness or the stated time '
    'complexity relies on dict/set average-case hashing), "benchmark_model" '
    "(string, the expected asymptotic running time reduced to a SINGLE "
    "variable n using the scaling note — e.g. a multi-variable complexity "
    'like O(n+m) or O(n*L) must be rewritten in terms of n alone by '
    "substituting how the other variable actually scales with n. Restricted "
    "to exactly this grammar: a product of n^a and (log n)^b, with a in "
    "{0, 0.5, 1, 1.5, 2, 3} and b in {0, 1}, written as one of: \"1\", "
    '"log n", "n", "n log n", "n^0.5", "n^0.5 log n", "n^1.5", '
    '"n^1.5 log n", "n^2", "n^2 log n", "n^3", "n^3 log n" — no other '
    'symbols, variables, or constants), "summary" (string, 1-2 sentences '
    'describing the approach), "notes" (string, optional remarks on '
    "idiomatic style or readability; empty string if none). "
    "When a problem statement is provided, treat every constraint it states "
    "(bounds on input size, value ranges, uniqueness guarantees, etc.) as a "
    "guaranteed precondition of the input, not an assumption made by the "
    "solution — relying on a stated constraint is correct and should not be "
    "flagged in notes as an unjustified assumption."
)


def iter_submissions(only_slug=None):
    if not SUBMISSIONS_ROOT.is_dir():
        return
    for slug_dir in sorted(SUBMISSIONS_ROOT.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        if only_slug and slug != only_slug:
            continue
        for path in sorted(slug_dir.iterdir()):
            m = SUBMISSION_RE.match(path.name)
            if not m:
                continue
            yield slug, int(m.group(1)), path


def golf_stats(source: str) -> dict:
    characters = len(source)
    byte_count = len(source.encode("utf-8"))
    non_blank_lines = sum(1 for line in source.splitlines() if line.strip())

    token_count = 0
    excluded = {
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
        tokenize.COMMENT,
    }
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type not in excluded:
                token_count += 1
    except tokenize.TokenizeError:
        token_count = None

    return {
        "characters": characters,
        "bytes": byte_count,
        "non_blank_lines": non_blank_lines,
        "tokens": token_count,
    }


def git_added_date(path: Path) -> str | None:
    rel = path.relative_to(REPO_ROOT)
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%aI", "-1", "--", str(rel)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    return out or None


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_object(text: str) -> str:
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


REQUIRED_FIELDS = (
    "time_average",
    "time_worst",
    "space",
    "hash_dependent",
    "benchmark_model",
    "summary",
)


def parse_complexity_response(raw_text: str) -> dict:
    candidate = extract_json_object(raw_text)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("model response is not a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"missing required field: {field}")
    data.setdefault("notes", "")
    return data


def build_user_prompt(
    slug: str, source: str, scaling_note: str | None, statement: str | None = None
) -> str:
    statement_line = ""
    if statement:
        truncated = statement[:STATEMENT_MAX_CHARS]
        statement_line = f"Problem statement (from NeetCode):\n{truncated}\n\n"
    scaling_line = f"Scaling: {scaling_note}\n\n" if scaling_note else ""
    return (
        f"Problem slug: {slug}\n\n{statement_line}{scaling_line}"
        f"Solution:\n```python\n{source}\n```"
    )


def call_moonshot(
    slug: str, source: str, scaling_note: str | None = None, statement: str | None = None
) -> dict:
    if not MOONSHOT_API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY is not set")

    user_prompt = build_user_prompt(slug, source, scaling_note, statement)
    payload = {
        "model": MOONSHOT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        f"{MOONSHOT_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MOONSHOT_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def get_complexity(
    slug: str,
    source: str,
    mock: bool,
    scaling_note: str | None = None,
    statement: str | None = None,
) -> tuple[dict | None, str | None]:
    if mock:
        return (
            {
                "time_average": "O(n)",
                "time_worst": "O(n)",
                "space": "O(n)",
                "hash_dependent": False,
                "benchmark_model": "n",
                "summary": "Mock analysis placeholder (no LLM call was made).",
                "notes": "",
            },
            None,
        )

    last_error = None
    for attempt in range(2):
        try:
            raw_text = call_moonshot(slug, source, scaling_note, statement)
            return parse_complexity_response(raw_text), None
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            IndexError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    last_error += f" — {exc.read().decode('utf-8', 'replace')[:500]}"
                except OSError:
                    pass
    return None, last_error


def analyze_submission(slug: str, number: int, path: Path, mock: bool) -> tuple[dict, bool]:
    source = path.read_text(encoding="utf-8")
    # generators.PROBLEMS may not have every slug (e.g. a problem added
    # without a benchmark generator yet); tolerate that and fall back to
    # no scaling note rather than raising.
    scaling_note = generators.PROBLEMS.get(slug, {}).get("scaling_note")

    # Mock mode never touches the network — including for the problem
    # statement fetch, which is otherwise best-effort and cached
    # in-process only (see scripts/statements.py; never written to disk).
    if mock:
        statement_slug, statement_text = None, None
    else:
        statement_slug, statement_text = statements.get_statement(slug)

    complexity, error = get_complexity(slug, source, mock, scaling_note, statement_text)

    record = {
        "file": str(path.relative_to(REPO_ROOT)),
        "slug": slug,
        "solved_at": git_added_date(path),
        "golf": golf_stats(source),
        "complexity": complexity,
        # The NeetCode slug (== this problem's directory name) when a
        # statement was actually fetched and included in the prompt, else
        # null — covers both "was a statement provided" and "which one"
        # in one field.
        "statement": statement_slug if statement_text else None,
        "analysis_version": ANALYSIS_VERSION,
        "model": MOONSHOT_MODEL if not mock else "mock",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        record["error"] = error
    return record, error is not None


def has_good_analysis(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict) or data.get("model") == "mock":
        # errored (null) and mock records are retried on later runs
        return False
    # records predating analysis_version (or below the current one) are
    # retried too — this is what forces every existing record through a
    # new prompt shape (e.g. the statement-inclusion change) at least once.
    if data.get("analysis_version", 1) < ANALYSIS_VERSION:
        return False
    complexity = data.get("complexity")
    if not isinstance(complexity, dict):
        return False
    # records predating the benchmark_model field are retried too, so the
    # field backfills onto existing analyses on the next run.
    return complexity.get("benchmark_model") is not None


def load_benchmarks(slug: str) -> dict:
    path = BENCHMARKS_ROOT / f"{slug}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get("submissions", {})


def build_summary():
    problems = {}
    if ANALYSIS_ROOT.is_dir():
        for slug_dir in sorted(ANALYSIS_ROOT.iterdir()):
            if not slug_dir.is_dir() or slug_dir == BENCHMARKS_ROOT:
                continue
            slug = slug_dir.name
            benchmarks = load_benchmarks(slug)
            submissions = []
            for json_path in sorted(slug_dir.glob("submission-*.json")):
                m = re.match(r"^submission-(\d+)\.json$", json_path.name)
                if not m:
                    continue
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                bench = benchmarks.get(f"submission-{m.group(1)}.py")
                if bench is not None:
                    data["benchmarks"] = bench
                submissions.append((int(m.group(1)), data))
            submissions.sort(key=lambda pair: pair[0])
            problems[slug] = {"submissions": [data for _, data in submissions]}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "problems": problems,
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def new_this_run_path() -> Path:
    override = os.environ.get(NEW_THIS_RUN_ENV)
    if override:
        return Path(override)
    return ANALYSIS_ROOT / ".new-this-run.json"


def write_new_this_run(new_records: list) -> None:
    """Write the machine-readable list of this run's newly-analyzed
    submissions for scripts/announce.py — always written (even empty), so a
    quiet run doesn't leave a stale list from a previous one. Never
    committed: see .gitignore."""
    out_path = new_this_run_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(new_records, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--only")
    args = parser.parse_args()

    had_error = False
    analyzed = 0
    new_records = []

    for slug, number, path in iter_submissions(args.only):
        out_dir = ANALYSIS_ROOT / slug
        out_path = out_dir / f"submission-{number}.json"
        if has_good_analysis(out_path):
            continue

        record, failed = analyze_submission(slug, number, path, args.mock)
        had_error = had_error or failed
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        analyzed += 1
        status = "ERROR" if failed else "ok"
        print(f"[{status}] {slug} submission-{number}")

        if not failed:
            new_records.append(
                {
                    "slug": slug,
                    "number": number,
                    "file": str(path.relative_to(REPO_ROOT)),
                    "analysis_file": str(out_path.relative_to(REPO_ROOT)),
                }
            )

    write_new_this_run(new_records)
    build_summary()
    print(f"Analyzed {analyzed} submission(s).")

    sys.exit(1 if had_error else 0)


if __name__ == "__main__":
    main()
