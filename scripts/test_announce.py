import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import announce
import statements


class HumanizeSlugTests(unittest.TestCase):
    def test_replaces_hyphens_and_titlecases(self):
        self.assertEqual(announce.humanize_slug("buy-and-sell-crypto"), "Buy And Sell Crypto")

    def test_single_word(self):
        self.assertEqual(announce.humanize_slug("anagram-groups"), "Anagram Groups")


class DifficultyColorTests(unittest.TestCase):
    def test_easy_is_green(self):
        self.assertEqual(announce.difficulty_color("Easy"), 0x57F287)

    def test_medium_is_yellow(self):
        self.assertEqual(announce.difficulty_color("Medium"), 0xFEE75C)

    def test_hard_is_red(self):
        self.assertEqual(announce.difficulty_color("Hard"), 0xED4245)

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(announce.difficulty_color(None), announce.DEFAULT_COLOR)
        self.assertEqual(announce.difficulty_color("Bogus"), announce.DEFAULT_COLOR)


class SpoilerFilenameTests(unittest.TestCase):
    def test_filename_is_spoilered_and_slug_and_number_qualified(self):
        name = announce.spoiler_filename("two-integer-sum", 1)
        self.assertEqual(name, "SPOILER_two-integer-sum-submission-1.png")
        self.assertTrue(name.startswith("SPOILER_"))


class FetchProblemMetaTests(unittest.TestCase):
    def test_uses_name_and_difficulty_when_present(self):
        with mock.patch.object(
            statements, "fetch_problem", return_value={"name": "Two Sum", "difficulty": "Easy"}
        ):
            meta = announce.fetch_problem_meta("two-integer-sum")
        self.assertEqual(meta, {"name": "Two Sum", "difficulty": "Easy"})

    def test_falls_back_to_humanized_slug_when_no_name(self):
        with mock.patch.object(statements, "fetch_problem", return_value={"difficulty": "Hard"}):
            meta = announce.fetch_problem_meta("two-integer-sum")
        self.assertEqual(meta["name"], "Two Integer Sum")
        self.assertEqual(meta["difficulty"], "Hard")

    def test_falls_back_fully_when_fetch_returns_none(self):
        with mock.patch.object(statements, "fetch_problem", return_value=None):
            meta = announce.fetch_problem_meta("is-palindrome")
        self.assertEqual(meta, {"name": "Is Palindrome", "difficulty": None})

    def test_falls_back_fully_when_fetch_raises(self):
        with mock.patch.object(statements, "fetch_problem", side_effect=RuntimeError("boom")):
            meta = announce.fetch_problem_meta("is-palindrome")
        self.assertEqual(meta, {"name": "Is Palindrome", "difficulty": None})


class BuildEmbedTests(unittest.TestCase):
    def _analysis(self, **overrides):
        analysis = {
            "complexity": {
                "time_average": "O(n)",
                "time_worst": "O(n^2)",
                "space": "O(1)",
            },
            "golf": {"tokens": 12, "non_blank_lines": 3, "characters": 80},
        }
        analysis.update(overrides)
        return analysis

    def test_embed_shape(self):
        embed = announce.build_embed({"name": "Two Sum", "difficulty": "Easy"})
        self.assertEqual(embed["title"], "Two Sum (Easy)")
        self.assertEqual(embed["url"], announce.PROJECT_URL)
        self.assertEqual(embed["color"], 0x57F287)

    def test_embed_carries_no_spoilers(self):
        # April's rule: nothing that hints at the approach may be visible
        # outside the spoilered image — no complexity, no length stats, and
        # (per discord-api-docs #1235) no attachment reference, which would
        # render the image unblurred inside the embed.
        embed = announce.build_embed({"name": "Two Sum", "difficulty": None})
        self.assertNotIn("image", embed)
        self.assertNotIn("fields", embed)
        self.assertEqual(embed["title"], "Two Sum")

    def test_stats_header_contents(self):
        header = announce.stats_header(self._analysis(), {"name": "Two Sum", "difficulty": "Easy"})
        self.assertIn("# Two Sum (Easy)", header)
        self.assertIn("O(n)", header)
        self.assertIn("O(n^2)", header)
        self.assertIn("12 tokens", header)
        self.assertTrue(all(l.startswith("#") or not l for l in header.splitlines()))

    def test_stats_header_tolerates_missing_complexity_and_golf(self):
        header = announce.stats_header({}, {"name": "X", "difficulty": None})
        self.assertIn("?", header)


class RenderSourceImageTests(unittest.TestCase):
    def test_returns_none_when_binary_missing(self):
        with mock.patch.object(announce, "FREEZE_BIN", "/nonexistent/freeze-binary-xyz"):
            result = announce.render_source_image(Path(__file__))
        self.assertIsNone(result)

    def test_returns_none_on_nonzero_exit(self):
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["freeze"]),
        ):
            result = announce.render_source_image(Path(__file__))
        self.assertIsNone(result)

    def test_returns_bytes_when_freeze_succeeds(self):
        def fake_run(cmd, check, capture_output, timeout):
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_bytes(b"\x89PNG-fake-bytes")
            return mock.Mock(returncode=0)

        with mock.patch("subprocess.run", side_effect=fake_run):
            result = announce.render_source_image(Path(__file__))
        self.assertEqual(result, b"\x89PNG-fake-bytes")

    def test_returns_none_when_output_missing_despite_success(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)):
            result = announce.render_source_image(Path(__file__))
        self.assertIsNone(result)


class MultipartEncodingTests(unittest.TestCase):
    def _parse(self, body: bytes, boundary: str):
        """Minimal multipart/form-data parser, sufficient to round-trip
        what encode_multipart produces (not a general RFC 2388 parser)."""
        delimiter = f"--{boundary}".encode("utf-8")
        raw_parts = body.split(delimiter)
        fields = {}
        files = {}
        for raw in raw_parts:
            raw = raw.strip(b"\r\n")
            if not raw or raw == b"--":
                continue
            header_blob, _, content = raw.partition(b"\r\n\r\n")
            headers = header_blob.decode("utf-8")
            disposition_match = re.search(r'name="([^"]+)"', headers)
            if not disposition_match:
                continue
            name = disposition_match.group(1)
            filename_match = re.search(r'filename="([^"]+)"', headers)
            if filename_match:
                files[name] = (filename_match.group(1), content)
            else:
                fields[name] = content.decode("utf-8")
        return fields, files

    def test_round_trip_without_image(self):
        payload = {"embeds": [{"title": "Two Sum"}]}
        body, content_type = announce.encode_multipart(payload, None, None)
        boundary = content_type.split("boundary=")[1]
        fields, files = self._parse(body, boundary)
        self.assertEqual(json.loads(fields["payload_json"]), payload)
        self.assertEqual(files, {})

    def test_round_trip_with_image(self):
        payload = {"embeds": [{"title": "Two Sum"}]}
        image_bytes = b"\x89PNG\r\n\x1a\nfake-binary-data\x00\xff"
        body, content_type = announce.encode_multipart(payload, "SPOILER_x.png", image_bytes)
        boundary = content_type.split("boundary=")[1]
        fields, files = self._parse(body, boundary)
        self.assertEqual(json.loads(fields["payload_json"]), payload)
        filename, content = files["files[0]"]
        self.assertEqual(filename, "SPOILER_x.png")
        self.assertEqual(content, image_bytes)


class PostWebhookTests(unittest.TestCase):
    def test_success_returns_true(self):
        fake_resp = mock.MagicMock()
        fake_resp.__enter__.return_value.read.return_value = b"{}"
        with mock.patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            ok = announce.post_webhook("https://discord.example/webhook", {"embeds": []}, None, "x.png")
        self.assertTrue(ok)
        mock_urlopen.assert_called_once()

    def test_url_error_returns_false_without_raising(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            ok = announce.post_webhook("https://discord.example/webhook", {"embeds": []}, None, "x.png")
        self.assertFalse(ok)


class MainQuietPathTests(unittest.TestCase):
    def setUp(self):
        sys.argv = ["announce.py"]

    def test_no_webhook_is_quiet_noop(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("DISCORD_WEBHOOK_URL", None)
            with mock.patch.object(announce, "load_new_records") as mock_load:
                result = announce.main()
        mock_load.assert_not_called()
        self.assertEqual(result, 0)

    def test_no_new_records_is_quiet_noop(self):
        with mock.patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.example/webhook"}):
            with mock.patch.object(announce, "load_new_records", return_value=[]):
                with mock.patch.object(announce, "post_webhook") as mock_post:
                    result = announce.main()
        mock_post.assert_not_called()
        self.assertEqual(result, 0)

    def test_missing_new_this_run_file_is_quiet_noop(self):
        with mock.patch.object(analyze, "new_this_run_path", return_value=Path("/nonexistent/x.json")):
            self.assertEqual(announce.load_new_records(), [])


class MainPostsForEachRecordTests(unittest.TestCase):
    def setUp(self):
        sys.argv = ["announce.py"]
        self.records = [
            {"slug": "a", "number": 0, "file": "f0.py", "analysis_file": "a0.json"},
            {"slug": "b", "number": 0, "file": "f1.py", "analysis_file": "a1.json"},
        ]

    def test_posts_once_per_new_record(self):
        with mock.patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.example/webhook"}):
            with mock.patch.object(announce, "load_new_records", return_value=self.records):
                with mock.patch.object(
                    announce, "build_announcement", return_value=({"embeds": []}, None, "x.png")
                ):
                    with mock.patch.object(announce, "post_webhook", return_value=True) as mock_post:
                        with mock.patch("time.sleep") as mock_sleep:
                            result = announce.main()
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()  # sleeps between, not after the last
        self.assertEqual(result, 0)

    def test_returns_nonzero_when_a_post_fails(self):
        with mock.patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.example/webhook"}):
            with mock.patch.object(announce, "load_new_records", return_value=self.records):
                with mock.patch.object(
                    announce, "build_announcement", return_value=({"embeds": []}, None, "x.png")
                ):
                    with mock.patch.object(announce, "post_webhook", return_value=False):
                        with mock.patch("time.sleep"):
                            result = announce.main()
        self.assertEqual(result, 1)


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_payload_and_never_posts(self):
        sys.argv = ["announce.py", "--dry-run"]
        records = [{"slug": "a", "number": 0, "file": "f0.py", "analysis_file": "a0.json"}]
        with mock.patch.dict("os.environ", {}, clear=False):
            with mock.patch.object(announce, "load_new_records", return_value=records):
                with mock.patch.object(
                    announce, "build_announcement", return_value=({"embeds": [{"title": "X"}]}, None, "x.png")
                ):
                    with mock.patch.object(announce, "post_webhook") as mock_post:
                        result = announce.main()
        mock_post.assert_not_called()
        self.assertEqual(result, 0)

    def test_dry_run_works_without_webhook_set(self):
        sys.argv = ["announce.py", "--dry-run"]
        import os

        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("DISCORD_WEBHOOK_URL", None)
            with mock.patch.object(announce, "load_new_records", return_value=[]):
                result = announce.main()
        self.assertEqual(result, 0)


class BuildAnnouncementTests(unittest.TestCase):
    def test_falls_back_to_no_image_when_freeze_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "Data Structures & Algorithms" / "sample-problem"
            source_dir.mkdir(parents=True)
            source_path = source_dir / "submission-0.py"
            source_path.write_text("def solve():\n    return 1\n")

            analysis_dir = tmp_path / "analysis" / "sample-problem"
            analysis_dir.mkdir(parents=True)
            analysis_path = analysis_dir / "submission-0.json"
            analysis_path.write_text(
                json.dumps(
                    {
                        "complexity": {"time_average": "O(1)", "time_worst": "O(1)", "space": "O(1)"},
                        "golf": {"tokens": 3, "non_blank_lines": 2, "characters": 30},
                    }
                )
            )

            record = {
                "slug": "sample-problem",
                "number": 0,
                "file": "Data Structures & Algorithms/sample-problem/submission-0.py",
                "analysis_file": "analysis/sample-problem/submission-0.json",
            }

            with mock.patch.object(announce, "REPO_ROOT", tmp_path):
                with mock.patch.object(announce, "render_source_image", return_value=None):
                    with mock.patch.object(
                        announce,
                        "fetch_problem_meta",
                        return_value={"name": "Sample Problem", "difficulty": "Easy"},
                    ):
                        payload, image_bytes, filename = announce.build_announcement(record)

        self.assertIsNone(image_bytes)
        self.assertEqual(filename, "SPOILER_sample-problem-submission-0.png")
        self.assertNotIn("image", payload["embeds"][0])
        self.assertEqual(payload["embeds"][0]["title"], "Sample Problem (Easy)")


if __name__ == "__main__":
    unittest.main()
