import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import statements


class ResolveTitleSlugTests(unittest.TestCase):
    def test_known_overrides(self):
        self.assertEqual(statements.resolve_title_slug("two-integer-sum"), "two-sum")
        self.assertEqual(
            statements.resolve_title_slug("buy-and-sell-crypto"),
            "best-time-to-buy-and-sell-stock",
        )
        self.assertEqual(statements.resolve_title_slug("is-anagram"), "valid-anagram")

    def test_fallback_uses_slug_as_is(self):
        self.assertEqual(
            statements.resolve_title_slug("longest-consecutive-sequence"),
            "longest-consecutive-sequence",
        )
        self.assertEqual(statements.resolve_title_slug("valid-sudoku"), "valid-sudoku")
        self.assertEqual(statements.resolve_title_slug("some-unmapped-slug"), "some-unmapped-slug")


class HtmlToTextTests(unittest.TestCase):
    def test_strips_tags(self):
        text = statements.html_to_text("<p>Hello <strong>world</strong></p>")
        self.assertEqual(text, "Hello world")

    def test_unescapes_entities(self):
        text = statements.html_to_text("<p>a &amp; b &lt; c</p>")
        self.assertEqual(text, "a & b < c")

    def test_preserves_paragraph_breaks(self):
        text = statements.html_to_text("<p>First.</p><p>Second.</p>")
        self.assertIn("First.\n\nSecond.", text)

    def test_keeps_constraints_section_intact(self):
        html_content = (
            "<p>Do the thing.</p>"
            "<p><strong>Constraints:</strong></p>"
            "<ul><li>1 &lt;= n &lt;= 10^5</li><li>-10^9 &lt;= nums[i] &lt;= 10^9</li></ul>"
        )
        text = statements.html_to_text(html_content)
        self.assertIn("Constraints:", text)
        self.assertIn("1 <= n <= 10^5", text)
        self.assertIn("-10^9 <= nums[i] <= 10^9", text)


class GetStatementTests(unittest.TestCase):
    def setUp(self):
        statements._CACHE.clear()
        self.addCleanup(statements._CACHE.clear)

    def test_successful_fetch_returns_title_slug_and_text(self):
        with mock.patch.object(
            statements,
            "fetch_question",
            return_value={"title": "Two Sum", "content": "<p>Body.</p>"},
        ) as mock_fetch:
            title_slug, text = statements.get_statement("two-integer-sum")
        mock_fetch.assert_called_once_with("two-sum")
        self.assertEqual(title_slug, "two-sum")
        self.assertEqual(text, "Body.")

    def test_premium_null_content_returns_none_text(self):
        with mock.patch.object(
            statements,
            "fetch_question",
            return_value={"title": "Encode and Decode Strings", "content": None},
        ):
            title_slug, text = statements.get_statement("string-encode-and-decode")
        self.assertEqual(title_slug, "encode-and-decode-strings")
        self.assertIsNone(text)

    def test_unresolvable_slug_returns_none_text(self):
        with mock.patch.object(statements, "fetch_question", return_value=None):
            title_slug, text = statements.get_statement("totally-unknown-slug")
        self.assertEqual(title_slug, "totally-unknown-slug")
        self.assertIsNone(text)

    def test_fetch_question_swallows_url_errors(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
            result = statements.fetch_question("two-sum")
        self.assertIsNone(result)

    def test_fetch_question_swallows_bad_json(self):
        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"not json"

        with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
            result = statements.fetch_question("two-sum")
        self.assertIsNone(result)

    def test_result_is_cached_in_process(self):
        with mock.patch.object(
            statements,
            "fetch_question",
            return_value={"title": "Two Sum", "content": "<p>Body.</p>"},
        ) as mock_fetch:
            statements.get_statement("two-integer-sum")
            statements.get_statement("two-integer-sum")
        mock_fetch.assert_called_once()

    def test_cache_is_in_memory_only_not_on_disk(self):
        # The module has no file-writing code path at all — this test
        # documents that constraint so a future change tripping it is
        # caught here.
        source = Path(statements.__file__).read_text(encoding="utf-8")
        for token in ("write_text", "import pathlib", "from pathlib", ".write("):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
