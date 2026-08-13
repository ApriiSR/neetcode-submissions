import json
import math
import random
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import benchmark
import generators

REPO_ROOT = Path(__file__).resolve().parent.parent


class GeneratorDeterminismTests(unittest.TestCase):
    def test_generate_is_deterministic_for_same_seed(self):
        for slug, spec in generators.PROBLEMS.items():
            first = spec["generate"](20, random.Random(42))
            second = spec["generate"](20, random.Random(42))
            self.assertEqual(first, second, slug)

    def test_generate_differs_for_different_seed(self):
        varying = False
        for slug, spec in generators.PROBLEMS.items():
            first = spec["generate"](20, random.Random(1))
            second = spec["generate"](20, random.Random(2))
            if first != second:
                varying = True
        self.assertTrue(varying)

    def test_adversarial_is_deterministic(self):
        for slug, spec in generators.PROBLEMS.items():
            if spec["adversarial"] is None:
                continue
            first = spec["adversarial"](30)
            second = spec["adversarial"](30)
            self.assertEqual(first, second, slug)


class IntCollisionPropertyTests(unittest.TestCase):
    def test_all_multiples_of_mersenne61_share_one_hash(self):
        values = generators._int_collisions(500)
        hashes = {hash(v) for v in values}
        self.assertEqual(hashes, {0})

    def test_collision_values_are_distinct(self):
        values = generators._int_collisions(500)
        self.assertEqual(len(set(values)), 500)

    def test_int_keyed_adversarial_generators_all_collide(self):
        for slug in (
            "duplicate-integer",
            "longest-consecutive-sequence",
            "top-k-elements-in-list",
            "two-integer-sum",
            "two-integer-sum-ii",
        ):
            spec = generators.PROBLEMS[slug]
            args = spec["adversarial"](200)
            nums = args[0]
            self.assertEqual(len({hash(v) for v in nums}), 1, slug)
            self.assertEqual(len(set(nums)), len(nums), slug)


class SlopeFittingTests(unittest.TestCase):
    def test_linear_timings_fit_slope_one(self):
        sizes = [256, 1024, 4096, 16384]
        times_ms = [0.01 * n for n in sizes]
        slope, r2 = benchmark.fit_loglog(sizes, times_ms)
        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_quadratic_timings_fit_slope_two(self):
        sizes = [256, 1024, 4096, 16384]
        times_ms = [0.01 * n * n for n in sizes]
        slope, r2 = benchmark.fit_loglog(sizes, times_ms)
        self.assertAlmostEqual(slope, 2.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_noisy_linear_timings_fit_near_slope_one(self):
        rng = random.Random(7)
        sizes = [256, 1024, 4096, 16384, 65536]
        times_ms = [0.01 * n * (1 + rng.uniform(-0.05, 0.05)) for n in sizes]
        slope, r2 = benchmark.fit_loglog(sizes, times_ms)
        self.assertAlmostEqual(slope, 1.0, delta=0.1)
        self.assertGreater(r2, 0.95)

    def test_single_point_returns_none(self):
        slope, r2 = benchmark.fit_loglog([256], [1.0])
        self.assertIsNone(slope)
        self.assertIsNone(r2)


class BestFitModelTests(unittest.TestCase):
    def test_linear_data_best_fit_is_n(self):
        sizes = [2**k for k in range(8, 17)]  # 256 .. 65536, 9 points
        times_ms = [0.01 * n for n in sizes]
        name, r2 = benchmark.fit_best_model(sizes, times_ms)
        self.assertEqual(name, "n")
        self.assertGreater(r2, 0.999)

    def test_nlogn_data_best_fit_is_n_log_n(self):
        sizes = [2**k for k in range(8, 17)]  # 256 .. 65536, 9 points
        times_ms = [0.01 * n * math.log(n) for n in sizes]
        name, r2 = benchmark.fit_best_model(sizes, times_ms)
        self.assertEqual(name, "n log n")
        self.assertGreater(r2, 0.999)

    def test_noisy_linear_data_best_fit_is_still_n(self):
        rng = random.Random(11)
        sizes = [2**k for k in range(8, 17)]
        times_ms = [0.01 * n * (1 + rng.uniform(-0.05, 0.05)) for n in sizes]
        name, r2 = benchmark.fit_best_model(sizes, times_ms)
        self.assertEqual(name, "n")
        self.assertGreater(r2, 0.95)

    def test_quadratic_data_best_fit_is_n_squared(self):
        sizes = [2**k for k in range(8, 17)]
        times_ms = [0.01 * n * n for n in sizes]
        name, r2 = benchmark.fit_best_model(sizes, times_ms)
        self.assertEqual(name, "n^2")
        self.assertGreater(r2, 0.999)

    def test_single_point_returns_none(self):
        name, r2 = benchmark.fit_best_model([256], [1.0])
        self.assertIsNone(name)
        self.assertIsNone(r2)

    def test_extra_candidate_can_win(self):
        # A K3 candidate matching the data exactly should beat the
        # built-in CANDIDATE_MODELS set when passed in as an extra.
        sizes = [2**k for k in range(8, 15)]
        times_ms = [0.01 * (n**0.5) for n in sizes]
        candidates = list(benchmark.CANDIDATE_MODELS) + [("n^0.5", lambda n: n**0.5)]
        name, r2 = benchmark.fit_best_model(sizes, times_ms, candidates)
        self.assertEqual(name, "n^0.5")
        self.assertGreater(r2, 0.999)


class BenchmarkModelGrammarTests(unittest.TestCase):
    def test_parses_bare_n(self):
        f = benchmark.parse_benchmark_model("n")
        self.assertEqual(f(1000), 1000)

    def test_parses_n_log_n(self):
        f = benchmark.parse_benchmark_model("n log n")
        self.assertAlmostEqual(f(1000), 1000 * math.log(1000))

    def test_parses_constant(self):
        f = benchmark.parse_benchmark_model("1")
        self.assertEqual(f(1000), 1.0)

    def test_parses_log_n(self):
        f = benchmark.parse_benchmark_model("log n")
        self.assertAlmostEqual(f(1000), math.log(1000))

    def test_parses_power(self):
        f = benchmark.parse_benchmark_model("n^2")
        self.assertEqual(f(10), 100)

    def test_parses_fractional_power(self):
        f = benchmark.parse_benchmark_model("n^1.5")
        self.assertAlmostEqual(f(100), 100**1.5)

    def test_parses_power_log(self):
        f = benchmark.parse_benchmark_model("n^2 log n")
        self.assertAlmostEqual(f(100), 100**2 * math.log(100))

    def test_is_case_and_whitespace_insensitive(self):
        f = benchmark.parse_benchmark_model("  N LOG N  ")
        self.assertAlmostEqual(f(1000), 1000 * math.log(1000))

    def test_rejects_out_of_grammar_exponent(self):
        self.assertIsNone(benchmark.parse_benchmark_model("n^4"))

    def test_rejects_free_text(self):
        self.assertIsNone(benchmark.parse_benchmark_model("O(n log n)"))
        self.assertIsNone(benchmark.parse_benchmark_model("linear"))
        self.assertIsNone(benchmark.parse_benchmark_model("n + m"))

    def test_rejects_non_string(self):
        self.assertIsNone(benchmark.parse_benchmark_model(None))
        self.assertIsNone(benchmark.parse_benchmark_model(42))

    def test_rejects_empty_string(self):
        self.assertIsNone(benchmark.parse_benchmark_model(""))


class LoadK3ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.patches = [
            mock.patch.object(analyze, "ANALYSIS_ROOT", self.tmp / "analysis"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def _write_analysis(self, slug, number, complexity):
        out_dir = self.tmp / "analysis" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"submission-{number}.json").write_text(
            json.dumps({"complexity": complexity}), encoding="utf-8"
        )

    def test_returns_none_when_no_analysis_exists(self):
        name, fn = benchmark.load_k3_model("some-slug", Path("submission-1.py"))
        self.assertIsNone(name)
        self.assertIsNone(fn)

    def test_loads_valid_model(self):
        self._write_analysis("some-slug", 1, {"benchmark_model": "n log n"})
        name, fn = benchmark.load_k3_model("some-slug", Path("submission-1.py"))
        self.assertEqual(name, "n log n")
        self.assertAlmostEqual(fn(1000), 1000 * math.log(1000))

    def test_ignores_invalid_grammar(self):
        self._write_analysis("some-slug", 1, {"benchmark_model": "O(n log n)"})
        name, fn = benchmark.load_k3_model("some-slug", Path("submission-1.py"))
        self.assertIsNone(name)
        self.assertIsNone(fn)

    def test_ignores_missing_benchmark_model_field(self):
        self._write_analysis("some-slug", 1, {"time_average": "O(n)"})
        name, fn = benchmark.load_k3_model("some-slug", Path("submission-1.py"))
        self.assertIsNone(name)
        self.assertIsNone(fn)


class RunLadderK3IntegrationTests(unittest.TestCase):
    def test_k3_model_fields_present_when_no_k3_model(self):
        result = benchmark.run_ladder(
            lambda nums: sum(nums),
            lambda n: ([1] * n,),
            [256, 512, 1024],
            10.0,
            time.perf_counter() + 10.0,
        )
        self.assertIsNone(result["k3_model"])
        self.assertIsNone(result["k3_model_r2"])
        self.assertIn("best_fit", result)

    def test_k3_model_included_as_candidate_and_can_win(self):
        sizes = [2**k for k in range(8, 15)]

        def fake_sort(nums):
            # deliberately not timed against real work; time_call measures
            # perf_counter around this, so we just need it to run.
            return sorted(nums)

        result = benchmark.run_ladder(
            fake_sort,
            lambda n: ([0] * n,),
            sizes,
            10.0,
            time.perf_counter() + 10.0,
            "n^0.5",
            lambda n: n**0.5,
        )
        self.assertEqual(result["k3_model"], "n^0.5")
        self.assertIsNotNone(result["k3_model_r2"])


class SolutionExecTests(unittest.TestCase):
    def test_loads_and_calls_a_real_submission(self):
        path = (
            REPO_ROOT
            / "Data Structures & Algorithms"
            / "duplicate-integer"
            / "submission-1.py"
        )
        solution = benchmark.load_solution(path)
        self.assertTrue(solution.hasDuplicate([1, 2, 3, 2]))
        self.assertFalse(solution.hasDuplicate([1, 2, 3]))

    def test_time_call_returns_nonnegative_seconds(self):
        path = (
            REPO_ROOT
            / "Data Structures & Algorithms"
            / "is-palindrome"
            / "submission-2.py"
        )
        solution = benchmark.load_solution(path)
        elapsed = benchmark.time_call(solution.isPalindrome, ("abcba",))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_copy_args_does_not_alias_lists(self):
        original = ([1, 2, 3],)
        copied = benchmark._copy_args(original)
        copied[0].append(4)
        self.assertEqual(original[0], [1, 2, 3])


class CorrectnessGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_error_guard_records_error_without_raising(self):
        path = self.tmp / "submission-broken.py"
        path.write_text(
            "class Solution:\n"
            "    def solve(self, nums: List[int]) -> int:\n"
            "        raise ValueError('boom')\n",
            encoding="utf-8",
        )
        spec = {
            "entry": "solve",
            "generate": lambda n, rng: ([1] * n,),
            "adversarial": None,
            "adversarial_note": None,
        }
        result = benchmark.benchmark_submission("broken-problem", spec, path)
        self.assertIn("error", result)
        self.assertIn("ValueError", result["error"])
        self.assertIn("boom", result["error"])

    def test_error_guard_triggers_from_adversarial_input_too(self):
        path = self.tmp / "submission-broken2.py"
        path.write_text(
            "class Solution:\n"
            "    def solve(self, nums: List[int]) -> int:\n"
            "        if len(nums) > 0 and nums[0] == -1:\n"
            "            raise ValueError('adversarial input broke this')\n"
            "        return sum(nums)\n",
            encoding="utf-8",
        )
        spec = {
            "entry": "solve",
            "generate": lambda n, rng: ([1] * n,),
            "adversarial": lambda n: ([-1] * n,),
            "adversarial_note": "n copies of -1",
        }
        result = benchmark.benchmark_submission("broken-problem-2", spec, path)
        self.assertIn("error", result)

    def test_good_submission_produces_timing_not_error(self):
        path = self.tmp / "submission-ok.py"
        path.write_text(
            "class Solution:\n"
            "    def solve(self, nums: List[int]) -> int:\n"
            "        return sum(nums)\n",
            encoding="utf-8",
        )
        spec = {
            "entry": "solve",
            "generate": lambda n, rng: ([1] * max(n, 1),),
            "adversarial": None,
            "adversarial_note": None,
        }
        result = benchmark.benchmark_submission("ok-problem", spec, path)
        self.assertNotIn("error", result)
        self.assertIn("slope", result)
        self.assertIn("sizes", result)


class HasGoodBenchmarkTests(unittest.TestCase):
    def test_error_record_is_not_good(self):
        self.assertFalse(benchmark.has_good_benchmark({"error": "boom"}))

    def test_normal_record_is_good(self):
        self.assertTrue(
            benchmark.has_good_benchmark(
                {"sizes": [1], "times_ms": [1.0], "slope": None, "r2": None, "adversarial": None}
            )
        )

    def test_missing_record_is_not_good(self):
        self.assertFalse(benchmark.has_good_benchmark(None))


if __name__ == "__main__":
    unittest.main()
