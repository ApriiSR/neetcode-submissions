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
# correct one. Bumped to 4 when the always-assume-hash-collisions prompt rule
# was scoped to unbounded input-controlled key spaces — constant key sets
# (fixed dict literals, small alphabets) were being wrongly rated O(n^2)
# worst case. Bumped to 5 when the prompt gained the tightest-bound rule for
# strategy-branching hybrids (report O(min(...)) instead of collapsing).
# Bumped to 6 when the collision rule was narrowed again, from "unbounded,
# input-controlled key space" to "hash values the input can actually choose":
# it was rating a set() of ListNode objects as O(n^2) worst case, but those
# use Python's default identity hash, so the addresses the allocator hands
# out — not the input — decide the hash values. Bumped to 7 when complexity
# moved to the GENERALIZED problem (NeetCode's numeric caps removed, domain
# kept — see generators.py's generalized_note) and the correctness field was
# added: a cap on n was being used to call solutions O(1), and a capped value
# range to rule out hash collisions, which left the page's claims resting on
# limits the code itself doesn't have. Bumped to 8 to state the runtime is
# CPython 3.14: a `-> tuple(int)` annotation got graded 'incorrect' for
# raising at definition time, which PEP 649 means it no longer does — the
# very reason the workflow pins 3.14.
ANALYSIS_VERSION = 8

# Per-request read timeout. Raised from 120s when the prompt grew (statement
# + generalized note + a longer system prompt all push response latency up)
# and a single slow response started tripping it.
REQUEST_TIMEOUT_SECONDS = 240

# Defensive cap on how much of a fetched problem statement goes into the
# prompt. NeetCode statements observed so far top out well under this; it
# exists to bound prompt size/cost if a future fetch returns something huge.
STATEMENT_MAX_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a precise algorithms reviewer. Given a problem slug, optionally "
    "its problem statement, a note on how that problem's input scales with "
    "n, and a Python solution, respond with STRICT JSON only (no markdown "
    "fences, no prose outside the JSON object) with exactly these fields: "
    '"time_average" (string, Big-O), "time_worst" (string, Big-O; when the '
    "solution hashes keys whose HASH VALUES are computed from unbounded, "
    "input-controlled data (ints, strings, or tuples built out of the input), "
    "the worst case must account for adversarial hash collisions degrading "
    "lookups to O(n). Two things do NOT qualify. First, keyed values from a "
    "bounded or constant set (a fixed dict literal, single characters from a "
    "fixed alphabet, a bounded computed signature) — collisions are bounded. "
    "Second, and this is the one most often gotten wrong: objects hashed by "
    "Python's DEFAULT identity hash, i.e. instances of a class that defines "
    "neither __hash__ nor __eq__, such as the ListNode/TreeNode objects "
    "handed to you by the problem. Their hash derives from the object's "
    "address, chosen by the allocator, so no input can steer it into "
    "collisions the way an attacker-chosen int can — putting nodes in a "
    "set()/dict is O(1) worst case, not O(n). Ask whether the INPUT can "
    "choose the hash values, not merely how many keys there are), "
    '"space" (string, Big-O), '
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
    'idiomatic style or readability; empty string if none), "correctness" '
    '(string, exactly one of "general", "neetcode-only", or "incorrect" — '
    "see the correctness rule below), and \"correctness_reason\" (string, "
    "one sentence naming the input that breaks it, or how it depends on a "
    "stated cap; empty string when \"general\"). "
    "When a solution branches between strategies (e.g. picks an algorithm "
    "based on input sizes), report the TIGHTEST correct bound, using min(...) "
    "or multi-variable forms if needed, rather than collapsing to a weaker "
    "single form — O(min(n + k, n log n)) is a stronger and better answer "
    "than O(n log n) for a guarded hybrid. "
    "Analyze COMPLEXITY against the GENERALIZED problem, not NeetCode's "
    "capped one. The generalized problem removes the statement's numeric "
    "limits — on input size and on the magnitude of values — while keeping "
    "everything that is part of the problem rather than a limit on it: the "
    "alphabet or character set, structural invariants (sortedness, "
    "uniqueness, a fixed-size board), and stated preconditions. A "
    "'Generalized problem' line gives the specifics when one is available; "
    "otherwise apply that rule to the stated constraints yourself. So never "
    "collapse a complexity to O(1) because the statement caps the input size "
    "— O(min(n, 1005)) = O(1) is exactly the wrong answer — and do not treat "
    "a capped value range as bounding the key space for hashing. A bound on "
    "something OTHER than input size that survives generalization (a "
    "26-letter alphabet, a 9x9 board) is a genuine constant and should be "
    "folded in. Arithmetic counts as unit cost even where generalizing "
    "removes a machine-word guarantee. "
    "Solutions run on CPython 3.14, where PEP 649 defers annotation "
    "evaluation: an annotation that would raise if evaluated eagerly "
    "(e.g. `-> tuple(int)`, which older Pythons would call at "
    "definition time) is simply never evaluated and cannot affect "
    "correctness. Say so in notes if it is worth a style remark, but "
    "never grade it as incorrect. "
    "When a 'Previous submission' is supplied, also report "
    '"relation_to_previous": "revision" if the new solution is the same '
    "approach as the previous one with only incidental changes — dead code "
    "removed, a line rewritten more neatly or more briefly, renaming, "
    'reformatting, a redundant lookup collapsed — or "new-approach" if it '
    "solves the problem by different means, uses a different data "
    "structure, or changes the shape of the algorithm. Judge the approach, "
    "not the diff size: a one-line change that swaps a list scan for a set "
    "lookup is a new approach, while ten lines of pure tidying is a "
    "revision. Omit the field entirely when no previous submission is "
    "given. "
    "Then judge CORRECTNESS on two levels, and report it in "
    '"correctness": "general" if the solution is right for every input to '
    'the generalized problem; "neetcode-only" if it is right for every '
    "input NeetCode's constraints permit but breaks once a cap is lifted "
    "(e.g. it hard-codes a maximum length, or indexes an array sized by a "
    'capped value); "incorrect" if some input NeetCode itself permits '
    "already breaks it — a delimiter drawn from the statement's own "
    "character set, an unhandled empty or duplicate case. Reserve "
    '"incorrect" for real counterexamples you can name in '
    '"correctness_reason", not for style or for relying on a stated '
    "precondition, which is legitimate. A solution being accepted by "
    "NeetCode is not evidence of correctness: its tests are not exhaustive, "
    'and "incorrect" here means the code is wrong on some permitted input '
    "regardless of what the judge said."
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
    "correctness",
)

# Anything else means the model invented a value; treated as a parse failure
# so the run retries rather than writing a status nothing can render.
VALID_CORRECTNESS = ("general", "neetcode-only", "incorrect")


def parse_complexity_response(raw_text: str) -> dict:
    candidate = extract_json_object(raw_text)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("model response is not a JSON object")
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"missing required field: {field}")
    if data["correctness"] not in VALID_CORRECTNESS:
        raise ValueError(f"unrecognized correctness: {data['correctness']!r}")
    data.setdefault("notes", "")
    data.setdefault("correctness_reason", "")
    return data


def build_user_prompt(
    slug: str,
    source: str,
    scaling_note: str | None,
    statement: str | None = None,
    generalized_note: str | None = None,
    previous_source: str | None = None,
) -> str:
    statement_line = ""
    if statement:
        truncated = statement[:STATEMENT_MAX_CHARS]
        statement_line = f"Problem statement (from NeetCode):\n{truncated}\n\n"
    scaling_line = f"Scaling: {scaling_note}\n\n" if scaling_note else ""
    generalized_line = (
        f"Generalized problem: {generalized_note}\n\n" if generalized_note else ""
    )
    previous_block = (
        f"Previous submission for this problem:\n```python\n{previous_source}\n```\n\n"
        if previous_source
        else ""
    )
    return (
        f"Problem slug: {slug}\n\n{statement_line}{generalized_line}{scaling_line}"
        f"{previous_block}Solution:\n```python\n{source}\n```"
    )


def call_moonshot(
    slug: str,
    source: str,
    scaling_note: str | None = None,
    statement: str | None = None,
    generalized_note: str | None = None,
    previous_source: str | None = None,
) -> dict:
    if not MOONSHOT_API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY is not set")

    user_prompt = build_user_prompt(
        slug, source, scaling_note, statement, generalized_note, previous_source
    )
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
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def get_complexity(
    slug: str,
    source: str,
    mock: bool,
    scaling_note: str | None = None,
    statement: str | None = None,
    generalized_note: str | None = None,
    previous_source: str | None = None,
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
                "correctness": "general",
                "correctness_reason": "",
            },
            None,
        )

    last_error = None
    for attempt in range(2):
        try:
            raw_text = call_moonshot(
                slug, source, scaling_note, statement, generalized_note, previous_source
            )
            return parse_complexity_response(raw_text), None
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            # socket.timeout is an alias of TimeoutError since 3.10 and is NOT a
            # URLError subclass, so it used to escape and kill the whole run —
            # every submission after the slow one lost, on one slow response.
            TimeoutError,
            OSError,
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


def previous_submission(slug: str, number: int) -> tuple[Path | None, dict | None]:
    """The highest-numbered submission below `number` that has an analysis
    record, with that record. NeetCode's sync never edits a file in place —
    re-submitting a problem writes a new submission-N+1.py — so this is the
    only handle on "what did the last attempt look like"."""
    best = None
    for found_slug, found_number, found_path in iter_submissions(slug):
        if found_number >= number:
            continue
        if best is None or found_number > best[0]:
            best = (found_number, found_path)
    if best is None:
        return None, None
    record_path = ANALYSIS_ROOT / slug / f"submission-{best[0]}.json"
    if not record_path.is_file():
        return best[1], None
    try:
        return best[1], json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return best[1], None


def _asymptotics(complexity) -> tuple | None:
    if not isinstance(complexity, dict):
        return None
    return (
        complexity.get("time_average"),
        complexity.get("time_worst"),
        complexity.get("space"),
    )


def settle_relation(complexity: dict | None, previous_record: dict | None) -> str | None:
    """Decide whether this submission supersedes the previous one.

    K3 judges whether the approach changed, since that is a judgement about
    intent that no diff ratio captures — but a change in asymptotics settles
    it outright, whatever K3 said. April's rule: tidying (a dead line
    removed, a line rewritten shorter) is a revision; anything that moves the
    complexity is a submission in its own right.
    """
    if not isinstance(complexity, dict):
        return None
    relation = complexity.get("relation_to_previous")
    if relation is None:
        return None
    if relation not in ("revision", "new-approach"):
        return "new-approach"
    if relation == "revision" and previous_record is not None:
        before = _asymptotics((previous_record or {}).get("complexity"))
        after = _asymptotics(complexity)
        if before is not None and after is not None and before != after:
            return "new-approach"
    return relation


def analyze_submission(slug: str, number: int, path: Path, mock: bool) -> tuple[dict, bool]:
    source = path.read_text(encoding="utf-8")
    # generators.PROBLEMS may not have every slug (e.g. a problem added
    # without a benchmark generator yet); tolerate that and fall back to
    # no scaling note rather than raising.
    scaling_note = generators.PROBLEMS.get(slug, {}).get("scaling_note")
    generalized_note = generators.PROBLEMS.get(slug, {}).get("generalized_note")

    # Mock mode never touches the network — including for the problem
    # statement fetch, which is otherwise best-effort and cached
    # in-process only (see scripts/statements.py; never written to disk).
    if mock:
        statement_slug, statement_text = None, None
    else:
        statement_slug, statement_text = statements.get_statement(slug)

    previous_path, previous_record = previous_submission(slug, number)
    previous_source = (
        previous_path.read_text(encoding="utf-8") if previous_path is not None else None
    )

    complexity, error = get_complexity(
        slug, source, mock, scaling_note, statement_text, generalized_note, previous_source
    )

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
    relation = settle_relation(complexity, previous_record)
    if relation is not None:
        record["relation_to_previous"] = relation
        if relation == "revision" and previous_path is not None:
            # The page hides a superseded submission and the announcer stays
            # quiet about the one superseding it: a tidy-up is not news.
            record["supersedes"] = previous_path.stem
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
        # An existing (stale/errored) record means this is a re-analysis —
        # e.g. a version bump — not a fresh solve; announce.py skips those.
        reanalysis = out_path.is_file()

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
                    "reanalysis": reanalysis,
                    # Carried here so announce.py can filter without reopening
                    # the record; absent when there was no previous submission.
                    "relation_to_previous": record.get("relation_to_previous"),
                }
            )

    write_new_this_run(new_records)
    build_summary()
    print(f"Analyzed {analyzed} submission(s).")

    sys.exit(1 if had_error else 0)


if __name__ == "__main__":
    main()
