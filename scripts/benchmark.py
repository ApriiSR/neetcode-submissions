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
ADVERSARIAL_SIZES = tuple(2**k for k in range(6, 17))  # 64 .. 65536, x2 each step
SIZE_CAP_SECONDS = 2.0
ADVERSARIAL_CAP_SECONDS = 1.5
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


def fit_best_model(sizes, times_ms):
    """Pick the candidate complexity model (see CANDIDATE_MODELS) whose
    log-space fit (slope fixed at 1, only the constant free) best
    explains the observed times, by residual sum of squares. Returns
    (label, r2) for the winner, or (None, None) with fewer than 2 points.
    """
    if len(sizes) < 2:
        return None, None
    log_t = [math.log(t) for t in times_ms]
    count = len(log_t)
    mean_log_t = sum(log_t) / count
    ss_tot = sum((y - mean_log_t) ** 2 for y in log_t)

    best_name, best_r2 = None, None
    for name, f in CANDIDATE_MODELS:
        residuals = [lt - math.log(f(n)) for lt, n in zip(log_t, sizes)]
        mean_resid = sum(residuals) / count
        ss_res = sum((r - mean_resid) ** 2 for r in residuals)
        r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        if best_r2 is None or r2 > best_r2:
            best_name, best_r2 = name, r2
    return best_name, best_r2


def run_ladder(fn, args_fn, sizes, cap_seconds, deadline):
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
    best_fit, best_fit_r2 = fit_best_model(sizes_out, times_out)
    return {
        "sizes": sizes_out,
        "times_ms": times_out,
        "slope": slope,
        "r2": r2,
        "best_fit": best_fit,
        "best_fit_r2": best_fit_r2,
    }


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
    )

    if spec["adversarial"] is not None:
        adversarial = run_ladder(
            entry,
            spec["adversarial"],
            ADVERSARIAL_SIZES,
            ADVERSARIAL_CAP_SECONDS,
            deadline,
        )
        adversarial["note"] = spec["adversarial_note"]
        result["adversarial"] = adversarial
    else:
        result["adversarial"] = None

    return result


def has_good_benchmark(record):
    return isinstance(record, dict) and "error" not in record


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
