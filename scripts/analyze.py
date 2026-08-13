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

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_ROOT = REPO_ROOT / "Data Structures & Algorithms"
ANALYSIS_ROOT = REPO_ROOT / "analysis"

# `or` fallbacks: the workflow exports these from repo vars, which arrive as
# empty strings when unset.
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1"
MOONSHOT_MODEL = os.environ.get("MOONSHOT_MODEL") or "kimi-k3"
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")

SUBMISSION_RE = re.compile(r"^submission-(\d+)\.py$")

SYSTEM_PROMPT = (
    "You are a precise algorithms reviewer. Given a problem slug and a Python "
    "solution, respond with STRICT JSON only (no markdown fences, no prose "
    "outside the JSON object) with exactly these fields: "
    '"time_average" (string, Big-O), "time_worst" (string, Big-O; if the '
    "solution uses a dict or set, the worst case must account for adversarial "
    "hash collisions degrading lookups to O(n)), \"space\" (string, Big-O), "
    '"hash_dependent" (boolean, true if correctness or the stated time '
    'complexity relies on dict/set average-case hashing), "summary" (string, '
    '1-2 sentences describing the approach), "notes" (string, optional '
    "remarks on idiomatic style or readability; empty string if none)."
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


def call_moonshot(slug: str, source: str) -> dict:
    if not MOONSHOT_API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY is not set")

    user_prompt = f"Problem slug: {slug}\n\nSolution:\n```python\n{source}\n```"
    payload = {
        "model": MOONSHOT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
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


def get_complexity(slug: str, source: str, mock: bool) -> tuple[dict | None, str | None]:
    if mock:
        return (
            {
                "time_average": "O(n)",
                "time_worst": "O(n)",
                "space": "O(n)",
                "hash_dependent": False,
                "summary": "Mock analysis placeholder (no LLM call was made).",
                "notes": "",
            },
            None,
        )

    last_error = None
    for attempt in range(2):
        try:
            raw_text = call_moonshot(slug, source)
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
    return None, last_error


def analyze_submission(slug: str, number: int, path: Path, mock: bool) -> tuple[dict, bool]:
    source = path.read_text(encoding="utf-8")
    complexity, error = get_complexity(slug, source, mock)

    record = {
        "file": str(path.relative_to(REPO_ROOT)),
        "slug": slug,
        "solved_at": git_added_date(path),
        "golf": golf_stats(source),
        "complexity": complexity,
        "model": MOONSHOT_MODEL if not mock else "mock",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        record["error"] = error
    return record, error is not None


def load_existing(out_path: Path) -> dict | None:
    if not out_path.is_file():
        return None
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_summary():
    problems = {}
    if ANALYSIS_ROOT.is_dir():
        for slug_dir in sorted(ANALYSIS_ROOT.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            submissions = []
            for json_path in sorted(slug_dir.glob("submission-*.json")):
                m = re.match(r"^submission-(\d+)\.json$", json_path.name)
                if not m:
                    continue
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--only")
    args = parser.parse_args()

    had_error = False
    analyzed = 0

    for slug, number, path in iter_submissions(args.only):
        out_dir = ANALYSIS_ROOT / slug
        out_path = out_dir / f"submission-{number}.json"
        if load_existing(out_path) is not None:
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

    build_summary()
    print(f"Analyzed {analyzed} submission(s).")

    sys.exit(1 if had_error else 0)


if __name__ == "__main__":
    main()
