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


class JsonParsingTests(unittest.TestCase):
    def test_plain_json(self):
        raw = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false, "summary": "s"}'
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["time_average"], "O(n)")
        self.assertEqual(data["notes"], "")

    def test_fenced_json(self):
        raw = '```json\n{"time_average": "O(n)", "time_worst": "O(n^2)", "space": "O(1)", "hash_dependent": true, "summary": "s", "notes": "n"}\n```'
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["time_worst"], "O(n^2)")
        self.assertTrue(data["hash_dependent"])

    def test_dirty_json_with_surrounding_prose(self):
        raw = (
            "Sure, here's the analysis:\n"
            '{"time_average": "O(n log n)", "time_worst": "O(n^2)", "space": "O(n)", '
            '"hash_dependent": false, "summary": "sorts then scans"}\n'
            "Let me know if you need more detail."
        )
        data = analyze.parse_complexity_response(raw)
        self.assertEqual(data["space"], "O(n)")

    def test_missing_required_field_raises(self):
        raw = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false}'
        with self.assertRaises(ValueError):
            analyze.parse_complexity_response(raw)

    def test_invalid_json_raises(self):
        raw = "not json at all"
        with self.assertRaises(json.JSONDecodeError):
            analyze.parse_complexity_response(raw)

    def test_retry_once_then_succeed(self):
        good = '{"time_average": "O(n)", "time_worst": "O(n)", "space": "O(1)", "hash_dependent": false, "summary": "s"}'
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

    def test_skips_already_analyzed_submission(self):
        out_dir = self.tmp / "analysis" / "sample-problem"
        out_dir.mkdir(parents=True)
        existing = {
            "file": "Data Structures & Algorithms/sample-problem/submission-0.py",
            "slug": "sample-problem",
            "solved_at": "2020-01-01T00:00:00-00:00",
            "golf": {"characters": 1, "bytes": 1, "non_blank_lines": 1, "tokens": 1},
            "complexity": {
                "time_average": "O(1)",
                "time_worst": "O(1)",
                "space": "O(1)",
                "hash_dependent": False,
                "summary": "preexisting",
                "notes": "",
            },
            "model": "kimi-k3",
            "analyzed_at": "2020-01-01T00:00:00+00:00",
        }
        (out_dir / "submission-0.json").write_text(json.dumps(existing))

        with mock.patch.object(analyze, "call_moonshot") as mock_call:
            sys.argv = ["analyze.py", "--mock"]
            with self.assertRaises(SystemExit):
                analyze.main()
            mock_call.assert_not_called()

        record = json.loads((out_dir / "submission-0.json").read_text())
        self.assertEqual(record["complexity"]["summary"], "preexisting")

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
