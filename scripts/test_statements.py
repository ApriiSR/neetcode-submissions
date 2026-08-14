import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import statements


def _response(data):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json

            return json.dumps(data).encode("utf-8")

    return _FakeResp()


class TruncateBeforeDetailsTests(unittest.TestCase):
    def test_cuts_at_first_details_block(self):
        description = (
            "Statement text.\n\n**Constraints:**\n* `1 <= n <= 100`\n\n"
            '<br>\n<details class="hint-accordion">\n'
            "    <summary>Topics</summary>\n</details>\n"
            '<details class="hint-accordion">\n'
            "    <summary>Recommended Time & Space Complexity</summary>\n"
            "    <p>You should aim for O(n) time.</p>\n</details>\n"
        )
        result = statements.truncate_before_details(description)
        self.assertNotIn("Recommended Time & Space Complexity", result)
        self.assertNotIn("<details", result)
        self.assertIn("Constraints:", result)
        self.assertIn("1 <= n <= 100", result)

    def test_no_details_block_returns_unchanged(self):
        description = "Just a statement with no accordions."
        self.assertEqual(statements.truncate_before_details(description), description)


class CleanMarkdownTests(unittest.TestCase):
    def test_strips_br_tags(self):
        text = statements.clean_markdown("Line one.\n<br>\n<br/>\nLine two.")
        self.assertNotIn("<br", text)
        self.assertIn("Line one.", text)
        self.assertIn("Line two.", text)

    def test_collapses_excess_blank_lines(self):
        text = statements.clean_markdown("First.\n\n\n\n\nSecond.")
        self.assertNotIn("\n\n\n", text)
        self.assertIn("First.\n\nSecond.", text)

    def test_preserves_markdown_and_constraints(self):
        description = (
            "Given `nums`, do the thing.\n\n**Constraints:**\n"
            "* `1 <= nums.length <= 1000`\n* `-10000 <= nums[i] <= 10000`\n"
        )
        text = statements.clean_markdown(description)
        self.assertIn("**Constraints:**", text)
        self.assertIn("`1 <= nums.length <= 1000`", text)
        self.assertIn("`-10000 <= nums[i] <= 10000`", text)


class FetchProblemTests(unittest.TestCase):
    def test_returns_data_dict_on_success(self):
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_response({"data": {"id": "two-integer-sum", "description": "x"}}),
        ):
            data = statements.fetch_problem("two-integer-sum")
        self.assertEqual(data["id"], "two-integer-sum")

    def test_unknown_slug_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=_response({"data": None})):
            self.assertIsNone(statements.fetch_problem("totally-bogus-slug"))

    def test_network_error_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
            self.assertIsNone(statements.fetch_problem("two-integer-sum"))

    def test_bad_json_returns_none(self):
        class _BadResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"not json"

        with mock.patch("urllib.request.urlopen", return_value=_BadResp()):
            self.assertIsNone(statements.fetch_problem("two-integer-sum"))

    def test_unexpected_top_level_shape_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=_response(["not", "a", "dict"])):
            self.assertIsNone(statements.fetch_problem("two-integer-sum"))

    def test_request_uses_exact_slug_as_problem_id(self):
        captured = {}

        def _fake_urlopen(request, timeout=None):
            import json

            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _response({"data": {"id": "two-integer-sum-ii", "description": "x"}})

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            statements.fetch_problem("two-integer-sum-ii")
        self.assertEqual(captured["body"], {"data": {"problemId": "two-integer-sum-ii"}})


class GetStatementTests(unittest.TestCase):
    def setUp(self):
        statements._CACHE.clear()
        self.addCleanup(statements._CACHE.clear)

    def test_successful_fetch_returns_slug_and_cleaned_text(self):
        description = (
            "Do the thing.\n\n**Constraints:**\n* `1 <= n <= 100`\n\n"
            '<br>\n<details class="hint-accordion">\n'
            "    <summary>Recommended Time & Space Complexity</summary>\n"
            "    <p>O(n) time expected.</p>\n</details>\n"
        )
        with mock.patch.object(
            statements, "fetch_problem", return_value={"description": description}
        ) as mock_fetch:
            slug, text = statements.get_statement("some-problem")
        mock_fetch.assert_called_once_with("some-problem")
        self.assertEqual(slug, "some-problem")
        self.assertIn("Constraints:", text)
        self.assertIn("1 <= n <= 100", text)
        self.assertNotIn("Recommended Time & Space Complexity", text)
        self.assertNotIn("O(n) time expected", text)

    def test_missing_slug_returns_none_text(self):
        with mock.patch.object(statements, "fetch_problem", return_value=None):
            slug, text = statements.get_statement("totally-bogus-slug")
        self.assertEqual(slug, "totally-bogus-slug")
        self.assertIsNone(text)

    def test_missing_description_field_returns_none_text(self):
        with mock.patch.object(statements, "fetch_problem", return_value={"id": "x"}):
            _, text = statements.get_statement("some-problem")
        self.assertIsNone(text)

    def test_empty_description_returns_none_text(self):
        with mock.patch.object(
            statements, "fetch_problem", return_value={"description": "   "}
        ):
            _, text = statements.get_statement("some-problem")
        self.assertIsNone(text)

    def test_description_that_is_only_a_details_block_returns_none_text(self):
        # Truncating before the first <details block can leave nothing —
        # that should degrade to None, not an empty-string "statement".
        with mock.patch.object(
            statements,
            "fetch_problem",
            return_value={"description": '<details class="hint-accordion">stuff</details>'},
        ):
            _, text = statements.get_statement("some-problem")
        self.assertIsNone(text)

    def test_result_is_cached_in_process(self):
        with mock.patch.object(
            statements, "fetch_problem", return_value={"description": "Body text."}
        ) as mock_fetch:
            statements.get_statement("some-problem")
            statements.get_statement("some-problem")
        mock_fetch.assert_called_once()

    def test_cache_is_in_memory_only_not_on_disk(self):
        source = Path(statements.__file__).read_text(encoding="utf-8")
        for token in ("write_text", "import pathlib", "from pathlib", ".write("):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
