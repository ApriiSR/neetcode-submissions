#!/usr/bin/env python3
"""Empirical scaling benchmarks + adversarial hash-collision demos.

For each submission of each scalable problem in `generators.PROBLEMS`,
times the entry method (best-of-3, `time.perf_counter`) across a
geometric size ladder, fits log(time) vs log(n) by least squares, and —
where the problem has an `adversarial()` generator — repeats on a
smaller ladder built from worst-case hash-collision input. Writes
`analysis/benchmarks/<slug>.json`, then regenerates `analysis/summary.json`
via `analyze.build_summary`, which merges the benchmark data in under a
`"benchmarks"` key per submission.

Requires PYTHONHASHSEED=0 (re-execs itself with it set if missing) so
runs are reproducible and any future string-keyed adversarial generator
has a fixed hash function to target.

Usage:
    python3 scripts/benchmark.py [--only SLUG] [--force]
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze
import generators

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = REPO_ROOT / "analysis" / "benchmarks"

DEFAULT_SIZES = tuple(2**k for k in range(8, 21))  # 256 .. 2**20, x2 each step
SIZE_CAP_SECONDS = 2.0
# The adversarial run uses the same ladder, so the two series are directly
# comparable point-for-point (and the progress page can plot them against the
# same x). It just gets a tighter per-size cap, since that's what decides
# where a genuinely quadratic case stops: an adversarial run that stays linear
# now walks the full ladder to 2**20 like the normal one, and one that blows
# up truncates itself, which is the "when possible" part.
ADVERSARIAL_CAP_SECONDS = 1.5

# Bumped when the measurement itself changes (ladder, caps, repeat counts) so
# stored records taken under the old regime are re-measured instead of sitting
# alongside new ones — the same role ANALYSIS_VERSION plays in analyze.py.
# Records predating this stamp have no "benchmark_version" and are stale by
# definition.
BENCHMARK_VERSION = 2
CORRECTNESS_N = 8
SEED = 20260813

# Above this size, drop from best-of-3 to best-of-2 to keep the denser,
# wider ladder's total wall time sane.
BEST_OF_2_THRESHOLD = 2**17

# Hard ceiling on total wall time (normal ladder + adversarial ladder
# combined) for a single submission, regardless of where the per-size
# caps above would otherwise let it run.
TOTAL_BUDGET_SECONDS = 30.0

# Candidate complexity models for best-fit classification. Each maps a
# short label to f(n); we fit log(t) = log(c) + log(f(n)) in log space
# (slope fixed at 1, only the constant c is free) and pick the model
# with the lowest residual, i.e. the highest R^2.
CANDIDATE_MODELS = (
    ("n", lambda n: n),
    ("n log n", lambda n: n * math.log(n)),
    ("n^1.5", lambda n: n**1.5),
    ("n^2", lambda n: n**2),
    ("n^3", lambda n: n**3),
)

# Grammar analyze.py's SYSTEM_PROMPT restricts K3's "benchmark_model" field
# to: a product of n^a and (log n)^b, a in {0, 0.5, 1, 1.5, 2, 3}, b in
# {0, 1}. parse_benchmark_model below accepts exactly the canonical
# spellings listed in that prompt and nothing else — anything outside the
# grammar (a hallucinated shape, a stale record predating this field) is
# rejected by returning None, and callers treat that as "no K3 model".
_BENCHMARK_MODEL_EXPONENTS = {0.5, 1, 1.5, 2, 3}
_BENCHMARK_MODEL_POWER_RE = re.compile(r"^n\^(\d+(?:\.\d+)?)$")
_BENCHMARK_MODEL_POWER_LOG_RE = re.compile(r"^n\^(\d+(?:\.\d+)?)\s+log\s+n$")


def parse_benchmark_model(text):
    if not isinstance(text, str):
        return None
    normalized = text.strip().lower()
    if normalized == "1":
        return lambda n: 1.0
    if normalized == "log n":
        return lambda n: math.log(n)
    if normalized == "n":
        return lambda n: n
    if normalized == "n log n":
        return lambda n: n * math.log(n)

    m = _BENCHMARK_MODEL_POWER_LOG_RE.match(normalized)
    if m:
        exp = float(m.group(1))
        return (lambda n, exp=exp: n**exp * math.log(n)) if exp in _BENCHMARK_MODEL_EXPONENTS else None

    m = _BENCHMARK_MODEL_POWER_RE.match(normalized)
    if m:
        exp = float(m.group(1))
        return (lambda n, exp=exp: n**exp) if exp in _BENCHMARK_MODEL_EXPONENTS else None

    return None


def load_solution(path):
    namespace = {"List": List, "Optional": Optional}
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["Solution"]()


def _copy_one(value):
    if isinstance(value, list):
        return [_copy_one(v) for v in value]
    return value


def _copy_args(args):
    return tuple(_copy_one(a) for a in args)


def time_call(fn, args, repeats=3):
    best = None
    for _ in range(repeats):
        call_args = _copy_args(args)
        start = time.perf_counter()
        fn(*call_args)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def fit_loglog(sizes, times_ms):
    if len(sizes) < 2:
        return None, None
    xs = [math.log(n) for n in sizes]
    ys = [math.log(t) for t in times_ms]
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, r2


def _fit_model_r2(sizes, times_ms, f):
    """R^2 of the log-space fit log(t) = log(c) + log(f(n)) (slope fixed at
    1, only the constant c free) against the observed (sizes, times_ms).
    """
    log_t = [math.log(t) for t in times_ms]
    count = len(log_t)
    mean_log_t = sum(log_t) / count
    ss_tot = sum((y - mean_log_t) ** 2 for y in log_t)
    residuals = [lt - math.log(f(n)) for lt, n in zip(log_t, sizes)]
    mean_resid = sum(residuals) / count
    ss_res = sum((r - mean_resid) ** 2 for r in residuals)
    return 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def fit_best_model(sizes, times_ms, candidates=CANDIDATE_MODELS):
    """Pick the candidate complexity model (see CANDIDATE_MODELS, plus any
    extra candidates passed in — e.g. a K3-supplied benchmark_model) whose
    log-space fit best explains the observed times, by R^2. Returns
    (label, r2) for the winner, or (None, None) with fewer than 2 points.
    """
    if len(sizes) < 2:
        return None, None
    best_name, best_r2 = None, None
    for name, f in candidates:
        r2 = _fit_model_r2(sizes, times_ms, f)
        if best_r2 is None or r2 > best_r2:
            best_name, best_r2 = name, r2
    return best_name, best_r2


def run_ladder(fn, args_fn, sizes, cap_seconds, deadline, k3_model_name=None, k3_model_fn=None):
    sizes_out, times_out = [], []
    for n in sizes:
        if time.perf_counter() >= deadline:
            break
        repeats = 2 if n > BEST_OF_2_THRESHOLD else 3
        args = args_fn(n)
        elapsed = time_call(fn, args, repeats=repeats)
        sizes_out.append(n)
        times_out.append(elapsed * 1000.0)
        if elapsed > cap_seconds:
            break
    slope, r2 = fit_loglog(sizes_out, times_out)

    candidates = list(CANDIDATE_MODELS)
    k3_model_r2 = None
    if k3_model_fn is not None and len(sizes_out) >= 2:
        k3_model_r2 = _fit_model_r2(sizes_out, times_out, k3_model_fn)
        candidates.append((k3_model_name, k3_model_fn))
    best_fit, best_fit_r2 = fit_best_model(sizes_out, times_out, candidates)

    return {
        "sizes": sizes_out,
        "times_ms": times_out,
        "slope": slope,
        "r2": r2,
        "best_fit": best_fit,
        "best_fit_r2": best_fit_r2,
        "k3_model": k3_model_name if k3_model_fn is not None else None,
        "k3_model_r2": k3_model_r2,
    }


def load_k3_model(slug, path):
    """Look up the K3-supplied `benchmark_model` for this submission's
    latest analysis record (if any) and parse it against the grammar.
    Returns (raw_string, callable) on a valid parse, or (None, None) if
    there's no analysis record yet, no benchmark_model field on it (an
    analysis predating that field), or the value doesn't match the
    grammar — all of which we treat the same way: no K3 candidate.
    """
    m = analyze.SUBMISSION_RE.match(path.name)
    if not m:
        return None, None
    analysis_path = analyze.ANALYSIS_ROOT / slug / f"submission-{m.group(1)}.json"
    if not analysis_path.is_file():
        return None, None
    try:
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    complexity = data.get("complexity") if isinstance(data, dict) else None
    if not isinstance(complexity, dict):
        return None, None
    raw = complexity.get("benchmark_model")
    fn = parse_benchmark_model(raw)
    if fn is None:
        return None, None
    return raw, fn


def benchmark_submission(slug, spec, path):
    rng = random.Random(f"{SEED}-{slug}-{path.name}")
    solution = load_solution(path)
    entry = getattr(solution, spec["entry"])

    small_args = spec["generate"](CORRECTNESS_N, rng)
    try:
        entry(*_copy_args(small_args))
        if spec["adversarial"] is not None:
            entry(*_copy_args(spec["adversarial"](CORRECTNESS_N)))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    k3_model_name, k3_model_fn = load_k3_model(slug, path)

    # Shared across the normal and adversarial ladders below: caps the
    # total wall time spent on this one submission regardless of how
    # generous the per-size caps are individually.
    deadline = time.perf_counter() + TOTAL_BUDGET_SECONDS

    result = run_ladder(
        entry,
        lambda n: spec["generate"](n, rng),
        DEFAULT_SIZES,
        SIZE_CAP_SECONDS,
        deadline,
        k3_model_name,
        k3_model_fn,
    )

    if spec["adversarial"] is not None:
        adversarial = run_ladder(
            entry,
            spec["adversarial"],
            DEFAULT_SIZES,
            ADVERSARIAL_CAP_SECONDS,
            deadline,
            k3_model_name,
            k3_model_fn,
        )
        adversarial["note"] = spec["adversarial_note"]
        result["adversarial"] = adversarial
    else:
        result["adversarial"] = None

    result["benchmark_version"] = BENCHMARK_VERSION
    return result


def has_good_benchmark(record):
    return (
        isinstance(record, dict)
        and "error" not in record
        and record.get("benchmark_version") == BENCHMARK_VERSION
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    BENCHMARKS_ROOT.mkdir(parents=True, exist_ok=True)
    had_error = False
    benchmarked = 0

    for slug, spec in generators.PROBLEMS.items():
        if not spec["scalable"]:
            continue
        if args.only and slug != args.only:
            continue

        out_path = BENCHMARKS_ROOT / f"{slug}.json"
        if out_path.is_file():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}
        submissions = existing.get("submissions", {})

        any_for_slug = False
        for found_slug, number, path in analyze.iter_submissions(slug):
            any_for_slug = True
            fname = path.name
            if not args.force and has_good_benchmark(submissions.get(fname)):
                continue

            record = benchmark_submission(slug, spec, path)
            submissions[fname] = record
            benchmarked += 1
            if "error" in record:
                had_error = True
                print(f"[ERROR] {slug} {fname}: {record['error']}")
            else:
                print(f"[ok] {slug} {fname} slope={record['slope']}")

        if not any_for_slug:
            continue

        out_path.write_text(
            json.dumps(
                {
                    "slug": slug,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "submissions": submissions,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    analyze.build_summary()
    print(f"Benchmarked {benchmarked} submission(s).")
    sys.exit(1 if had_error else 0)


if __name__ == "__main__":
    # re-exec, not os.environ: CPython only reads PYTHONHASHSEED at startup.
    # Guarded under __main__ so importing this module (e.g. from tests) never
    # replaces the importing process.
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        os.execvpe(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
            env,
        )
    main()
