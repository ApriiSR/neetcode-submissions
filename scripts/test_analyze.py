import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import statements


class GolfStatsTests(unittest.TestCase):
    def test_basic_counts(self):
        source = "x = 1\ny = 2\n\n"
        stats = analyze.golf_stats(source)
        self.assertEqual(stats["characters"], len(source))
        self.assertEqual(stats["bytes"], len(source.encode("utf-8")))
        self.assertEqual(stats["non_blank_lines"], 2)

    def test_token_count_excludes_structural_tokens(self):
        source = "x = 1\n"
        stats = analyze.golf_stats(source)
        # NAME, OP, NUMBER = 3 significant tokens; NEWLINE/ENDMARKER excluded
        self.assertEqual(stats["tokens"], 3)

    def test_token_count_excludes_comments(self):
        with_comment = "x = 1  # comment\n"
        without_comment = "x = 1\n"
        self.assertEqual(
            analyze.golf_stats(with_comment)["tokens"],
            analyze.golf_stats(without_comment)["tokens"],
        )

    def test_multibyte_characters_count_more_bytes_than_chars(self):
        source = "x = 'héllo'\n"
        stats = analyze.golf_stats(source)
        self.assertGreater(stats["bytes"], stats["characters"])


class SettleRelationTests(unittest.TestCase):
    """April's rule: tidying is a revision, changed asymptotics never is."""

    PREV = {"complexity": {"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)"}}

    def test_no_relation_when_the_model_did_not_report_one(self):
        # No previous submission was supplied, so the field is absent.
        self.assertIsNone(analyze.settle_relation({"time_average": "O(n)"}, None))

    def test_revision_survives_when_asymptotics_match(self):
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
            "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, self.PREV), "revision")

    def test_changed_asymptotics_overrides_a_revision_verdict(self):
        # K3 called it a tidy-up, but the complexity moved — that is a
        # submission in its own right regardless of how small the diff was.
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n^2)", "space": "O(1)",
            "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, self.PREV), "new-approach")

    def test_changed_space_also_overrides(self):
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(n)",
            "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, self.PREV), "new-approach")

    def test_changed_correctness_overrides_a_revision_verdict(self):
        # A fix must not quietly delete the broken version it supersedes.
        previous = {
            "complexity": {
                "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
                "correctness": "incorrect",
            }
        }
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
            "correctness": "general", "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, previous), "new-approach")

    def test_same_correctness_stays_a_revision(self):
        previous = {
            "complexity": {
                "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
                "correctness": "incorrect",
            }
        }
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
            "correctness": "incorrect", "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, previous), "revision")

    def test_missing_correctness_on_either_side_does_not_force(self):
        # Records predating the field can't be compared; don't invent a split.
        previous = {"complexity": {"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)"}}
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
            "correctness": "general", "relation_to_previous": "revision",
        }
        self.assertEqual(analyze.settle_relation(complexity, previous), "revision")

    def test_new_approach_is_left_alone(self):
        complexity = {
            "time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)",
            "relation_to_previous": "new-approach",
        }
        self.assertEqual(analyze.settle_relation(complexity, self.PREV), "new-approach")

    def test_unrecognized_relation_falls_back_to_new_approach(self):
        # Never silently swallow a submission on a value we don't understand.
        complexity = {"relation_to_previous": "sort of similar"}
        self.assertEqual(analyze.settle_relation(complexity, self.PREV), "new-approach")

    def test_missing_previous_record_keeps_the_models_verdict(self):
        complexity = {"relation_to_previous": "revision"}
        self.assertEqual(analyze.settle_relation(complexity, None), "revision")


class JsonParsingTests(unittest.TestCase):
    def test_plain_json(self):
        raw = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false, "benchmark_model": "n", "summary": "s", "correctness": "general"}'
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["time_average"], "O(n)")
        self.assertEqual(data["notes"], "")

    def test_fenced_json(self):
        raw = '```json\n{"time_average": "O(n)", "time_worst": "O(n^2)", "space": "O(1)", "hash_dependent": true, "benchmark_model": "n^2", "summary": "s", "notes": "n", "correctness": "incorrect", "correctness_reason": "breaks on an empty list"}\n```'
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["time_worst"], "O(n^2)")
        self.assertTrue(data["hash_dependent"])

    def test_dirty_json_with_surrounding_prose(self):
        raw = (
            "Sure, here's the analysis:\n"
            '{"time_average": "O(n log n)", "time_worst": "O(n^2)", "space": "O(n)", '
            '"hash_dependent": false, "benchmark_model": "n log n", "summary": "sorts then scans", "correctness": "general"}\n'
            "Let me know if you need more detail."
        )
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["space"], "O(n)")

    def test_unrecognized_correctness_is_a_parse_failure(self):
        # A status nothing can render is worse than a retry.
        raw = ('{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", '
               '"hash_dependent": false, "benchmark_model": "n", "summary": "s", '
               '"correctness": "probably fine"}')
        with self.assertRaises(ValueError):
            analyze.parse_complexity_response(raw)

    def test_correctness_reason_defaults_to_empty(self):
        raw = ('{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", '
               '"hash_dependent": false, "benchmark_model": "n", "summary": "s", '
               '"correctness": "general"}')
        self.assertEqual(analyze.parse_complexity_response(raw)["correctness_reason"], "")

    def test_missing_required_field_raises(self):
        raw = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false}'
        with self.assertRaises(ValueError):
            analyze.parse_complexity_response(raw)

    def test_read_timeout_is_retried_not_fatal(self):
        # One slow response used to abort the whole pass, losing every
        # submission after it.
        good = ('{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", '
                '"hash_dependent": false, "benchmark_model": "n", "summary": "s", '
                '"correctness": "general"}')
        with mock.patch.object(analyze, "MOONSHOT_API_KEY", "fake-key"):
            with mock.patch.object(
                analyze, "call_moonshot", side_effect=[TimeoutError("read timed out"), good]
            ):
                data, error = analyze.get_complexity("slug", "src", mock=False)
        self.assertIsNone(error)
        self.assertEqual(data["time_average"], "O(n)")

    def test_persistent_timeout_reports_an_error_instead_of_raising(self):
        with mock.patch.object(analyze, "MOONSHOT_API_KEY", "fake-key"):
            with mock.patch.object(
                analyze, "call_moonshot", side_effect=TimeoutError("read timed out")
            ):
                data, error = analyze.get_complexity("slug", "src", mock=False)
        self.assertIsNone(data)
        self.assertIn("TimeoutError", error)

    def test_invalid_json_raises(self):
        raw = "not json at all"
        with self.assertRaises(json.JSONDecodeError):
            analyze.parse_complexity_response(raw)

    def test_retry_once_then_succeed(self):
        good = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false, "benchmark_model": "n", "summary": "s", "correctness": "general"}'
        with mock.patch.object(analyze, "MOONSHOT_API_KEY", "fake-key"):
            with mock.patch.object(
                analyze, "call_moonshot", side_effect=["not json", good]
            ) as mock_call:
                data, error = analyze.get_complexity("slug", "code", mock=False)
        self.assertIsNone(error)
        self.assertEqual(data["time_average"], "O(n)")
        self.assertEqual(mock_call.call_count, 2)

    def test_persistent_failure_returns_error(self):
        with mock.patch.object(analyze, "MOONSHOT_API_KEY", "fake-key"):
            with mock.patch.object(
                analyze, "call_moonshot", side_effect=["nope", "still nope"]
            ):
                data, error = analyze.get_complexity("slug", "code", mock=False)
        self.assertIsNone(data)
        self.assertIsNotNone(error)

    def test_mock_mode_skips_api(self):
        with mock.patch.object(analyze, "call_moonshot") as mock_call:
            data, error = analyze.get_complexity("slug", "code", mock=True)
        mock_call.assert_not_called()
        self.assertIsNone(error)
        self.assertEqual(data["summary"], "Mock analysis placeholder (no LLM call was made).")
        self.assertEqual(data["benchmark_model"], "n")


class BuildUserPromptTests(unittest.TestCase):
    def test_includes_scaling_note_when_given(self):
        prompt = analyze.build_user_prompt("some-slug", "code", "n = array length")
        self.assertIn("Scaling: n = array length", prompt)
        self.assertIn("some-slug", prompt)
        self.assertIn("code", prompt)

    def test_omits_scaling_line_when_none(self):
        prompt = analyze.build_user_prompt("some-slug", "code", None)
        self.assertNotIn("Scaling:", prompt)

    def test_includes_statement_when_given(self):
        prompt = analyze.build_user_prompt(
            "some-slug", "code", None, statement="1 <= nums.length <= 10^5"
        )
        self.assertIn("Problem statement (from NeetCode):", prompt)
        self.assertIn("1 <= nums.length <= 10^5", prompt)
        # statement comes before the solution
        self.assertLess(prompt.index("Problem statement"), prompt.index("Solution:"))

    def test_omits_statement_line_when_none(self):
        prompt = analyze.build_user_prompt("some-slug", "code", None, statement=None)
        self.assertNotIn("Problem statement", prompt)

    def test_omits_statement_line_when_empty_string(self):
        prompt = analyze.build_user_prompt("some-slug", "code", None, statement="")
        self.assertNotIn("Problem statement", prompt)

    def test_statement_truncated_defensively(self):
        huge = "x" * 20000
        prompt = analyze.build_user_prompt("some-slug", "code", None, statement=huge)
        self.assertLessEqual(
            prompt.count("x"), analyze.STATEMENT_MAX_CHARS + 10  # slack for other x's, there are none
        )
        self.assertNotIn("x" * (analyze.STATEMENT_MAX_CHARS + 1), prompt)


class HasGoodAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _record(self, **complexity_overrides):
        complexity = {
            "time_average": "O(n)",
            "time_worst": "O(n)",
            "space": "O(n)",
            "hash_dependent": False,
            "benchmark_model": "n",
            "summary": "s",
            "notes": "",
        }
        complexity.update(complexity_overrides)
        return {
            "complexity": complexity,
            "model": "kimi-k3",
            "analysis_version": analyze.ANALYSIS_VERSION,
        }

    def test_missing_file_is_not_good(self):
        self.assertFalse(analyze.has_good_analysis(self.tmp / "nope.json"))

    def test_complete_record_is_good(self):
        path = self._write("ok.json", self._record())
        self.assertTrue(analyze.has_good_analysis(path))

    def test_missing_benchmark_model_key_is_not_good(self):
        record = self._record()
        del record["complexity"]["benchmark_model"]
        path = self._write("no-bm.json", record)
        self.assertFalse(analyze.has_good_analysis(path))

    def test_null_benchmark_model_is_not_good(self):
        path = self._write("null-bm.json", self._record(benchmark_model=None))
        self.assertFalse(analyze.has_good_analysis(path))

    def test_mock_model_is_not_good(self):
        record = self._record()
        record["model"] = "mock"
        path = self._write("mock.json", record)
        self.assertFalse(analyze.has_good_analysis(path))

    def test_null_complexity_is_not_good(self):
        path = self._write(
            "errored.json",
            {"complexity": None, "model": "kimi-k3", "analysis_version": analyze.ANALYSIS_VERSION},
        )
        self.assertFalse(analyze.has_good_analysis(path))

    def test_missing_analysis_version_is_not_good(self):
        # records written before this field existed (analysis_version 1,
        # implicit) — must be retried so the statement-inclusion prompt
        # change reaches them.
        record = self._record()
        del record["analysis_version"]
        path = self._write("no-version.json", record)
        self.assertFalse(analyze.has_good_analysis(path))

    def test_old_analysis_version_is_not_good(self):
        record = self._record()
        record["analysis_version"] = 1
        path = self._write("old-version.json", record)
        self.assertFalse(analyze.has_good_analysis(path))


class RepoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=self.tmp, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.tmp, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.tmp, check=True)

        self.slug_dir = self.tmp / "Data Structures & Algorithms" / "sample-problem"
        self.slug_dir.mkdir(parents=True)
        sub_path = self.slug_dir / "submission-0.py"
        sub_path.write_text("def solve():\n    return 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=self.tmp, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Add: sample-problem - submission-0"], cwd=self.tmp, check=True)

        self.patches = [
            mock.patch.object(analyze, "REPO_ROOT", self.tmp),
            mock.patch.object(
                analyze, "SUBMISSIONS_ROOT", self.tmp / "Data Structures & Algorithms"
            ),
            mock.patch.object(analyze, "ANALYSIS_ROOT", self.tmp / "analysis"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_git_added_date_returns_iso_string(self):
        path = self.slug_dir / "submission-0.py"
        date = analyze.git_added_date(path)
        self.assertIsNotNone(date)
        self.assertRegex(date, r"^\d{4}-\d{2}-\d{2}T")

    def test_end_to_end_mock_run_creates_per_submission_and_summary(self):
        sys.argv = ["analyze.py", "--mock"]
        with self.assertRaises(SystemExit) as ctx:
            analyze.main()
        self.assertEqual(ctx.exception.code, 0)

        out_path = self.tmp / "analysis" / "sample-problem" / "submission-0.json"
        self.assertTrue(out_path.is_file())
        record = json.loads(out_path.read_text())
        self.assertEqual(record["slug"], "sample-problem")
        self.assertEqual(record["model"], "mock")
        self.assertIsNotNone(record["complexity"])

        summary_path = self.tmp / "analysis" / "summary.json"
        self.assertTrue(summary_path.is_file())
        summary = json.loads(summary_path.read_text())
        self.assertIn("sample-problem", summary["problems"])
        self.assertEqual(len(summary["problems"]["sample-problem"]["submissions"]), 1)

    def _existing_record(self, **complexity_overrides):
        complexity = {
            "time_average": "O(1)",
            "time_worst": "O(1)",
            "space": "O(1)",
            "hash_dependent": False,
            "benchmark_model": "1",
            "summary": "preexisting",
            "notes": "",
        }
        complexity.update(complexity_overrides)
        return {
            "file": "Data Structures & Algorithms/sample-problem/submission-0.py",
            "slug": "sample-problem",
            "solved_at": "2020-01-01T00:00:00-00:00",
            "golf": {"characters": 1, "bytes": 1, "non_blank_lines": 1, "tokens": 1},
            "complexity": complexity,
            "statement": None,
            "analysis_version": analyze.ANALYSIS_VERSION,
            "model": "kimi-k3",
            "analyzed_at": "2020-01-01T00:00:00+00:00",
        }

    def test_skips_already_analyzed_submission(self):
        out_dir = self.tmp / "analysis" / "sample-problem"
        out_dir.mkdir(parents=True)
        existing = self._existing_record()
        (out_dir / "submission-0.json").write_text(json.dumps(existing))

        with mock.patch.object(analyze, "call_moonshot") as mock_call:
            sys.argv = ["analyze.py", "--mock"]
            with self.assertRaises(SystemExit):
                analyze.main()
            mock_call.assert_not_called()

        record = json.loads((out_dir / "submission-0.json").read_text())
        self.assertEqual(record["complexity"]["summary"], "preexisting")

    def test_missing_benchmark_model_triggers_reanalysis(self):
        out_dir = self.tmp / "analysis" / "sample-problem"
        out_dir.mkdir(parents=True)
        existing = self._existing_record()
        del existing["complexity"]["benchmark_model"]
        (out_dir / "submission-0.json").write_text(json.dumps(existing))

        sys.argv = ["analyze.py", "--mock"]
        with self.assertRaises(SystemExit):
            analyze.main()

        # backfilled by re-analysis (mock, in this test) rather than skipped
        record = json.loads((out_dir / "submission-0.json").read_text())
        self.assertEqual(record["model"], "mock")
        self.assertEqual(record["complexity"]["benchmark_model"], "n")

    def test_old_analysis_version_triggers_reanalysis(self):
        out_dir = self.tmp / "analysis" / "sample-problem"
        out_dir.mkdir(parents=True)
        existing = self._existing_record()
        del existing["analysis_version"]  # simulates a pre-version-2 record
        (out_dir / "submission-0.json").write_text(json.dumps(existing))

        sys.argv = ["analyze.py", "--mock"]
        with self.assertRaises(SystemExit):
            analyze.main()

        record = json.loads((out_dir / "submission-0.json").read_text())
        self.assertEqual(record["model"], "mock")
        self.assertEqual(record["analysis_version"], analyze.ANALYSIS_VERSION)
        self.assertEqual(record["complexity"]["summary"], "Mock analysis placeholder (no LLM call was made).")

    def test_mock_run_includes_new_fields_without_network_fetch(self):
        sys.argv = ["analyze.py", "--mock"]
        with mock.patch.object(statements, "get_statement") as mock_get_statement:
            with self.assertRaises(SystemExit):
                analyze.main()
        mock_get_statement.assert_not_called()

        out_path = self.tmp / "analysis" / "sample-problem" / "submission-0.json"
        record = json.loads(out_path.read_text())
        self.assertIn("statement", record)
        self.assertIsNone(record["statement"])
        self.assertEqual(record["analysis_version"], analyze.ANALYSIS_VERSION)

    def test_new_this_run_lists_newly_analyzed_submission(self):
        sys.argv = ["analyze.py", "--mock"]
        with self.assertRaises(SystemExit):
            analyze.main()

        new_this_run_path = self.tmp / "analysis" / ".new-this-run.json"
        self.assertTrue(new_this_run_path.is_file())
        records = json.loads(new_this_run_path.read_text())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["slug"], "sample-problem")
        self.assertEqual(records[0]["number"], 0)
        self.assertEqual(
            records[0]["file"], "Data Structures & Algorithms/sample-problem/submission-0.py"
        )
        self.assertEqual(
            records[0]["analysis_file"], "analysis/sample-problem/submission-0.json"
        )

    def test_new_this_run_is_empty_and_overwritten_when_nothing_analyzed(self):
        out_dir = self.tmp / "analysis" / "sample-problem"
        out_dir.mkdir(parents=True)
        (out_dir / "submission-0.json").write_text(json.dumps(self._existing_record()))

        # A stale list from a previous run should not survive an empty run.
        new_this_run_path = self.tmp / "analysis" / ".new-this-run.json"
        new_this_run_path.parent.mkdir(parents=True, exist_ok=True)
        new_this_run_path.write_text(json.dumps([{"slug": "stale"}]))

        sys.argv = ["analyze.py", "--mock"]
        with self.assertRaises(SystemExit):
            analyze.main()

        records = json.loads(new_this_run_path.read_text())
        self.assertEqual(records, [])

    def test_new_this_run_excludes_errored_submissions(self):
        with mock.patch.object(analyze, "get_complexity", return_value=(None, "boom")):
            sys.argv = ["analyze.py", "--mock"]
            with self.assertRaises(SystemExit):
                analyze.main()

        new_this_run_path = self.tmp / "analysis" / ".new-this-run.json"
        records = json.loads(new_this_run_path.read_text())
        self.assertEqual(records, [])

    def test_new_this_run_path_env_override(self):
        override = self.tmp / "elsewhere.json"
        with mock.patch.dict("os.environ", {analyze.NEW_THIS_RUN_ENV: str(override)}):
            sys.argv = ["analyze.py", "--mock"]
            with self.assertRaises(SystemExit):
                analyze.main()

        self.assertTrue(override.is_file())
        self.assertFalse((self.tmp / "analysis" / ".new-this-run.json").exists())

    def test_only_flag_filters_slug(self):
        other_dir = self.tmp / "Data Structures & Algorithms" / "other-problem"
        other_dir.mkdir(parents=True)
        (other_dir / "submission-0.py").write_text("x = 1\n", encoding="utf-8")

        sys.argv = ["analyze.py", "--mock", "--only", "sample-problem"]
        with self.assertRaises(SystemExit):
            analyze.main()

        self.assertTrue(
            (self.tmp / "analysis" / "sample-problem" / "submission-0.json").is_file()
        )
        self.assertFalse((self.tmp / "analysis" / "other-problem").exists())


if __name__ == "__main__":
    unittest.main()
