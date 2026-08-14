"""Fetch NeetCode/LeetCode problem statements for use in complexity analysis.

`scripts/analyze.py` asks K3 to review a solution's complexity, but the
solution's source code alone doesn't carry the problem's stated
constraints (e.g. "1 <= nums.length <= 10^5") — so K3 sometimes flags a
correct reliance on those constraints as an "unjustified assumption".
Fetching the actual problem text and including it in the prompt fixes
that.

NeetCode itself was investigated first (see README's "Automated
analysis" section for the writeup) and does not expose a usable public
data source: its problem pages are an Angular SPA and the HTML/API
responses observed while investigating this module carried no problem
content, only the app shell. LeetCode's GraphQL endpoint does, and most
NeetCode problems map directly onto a LeetCode problem, so that's the
source used here.

Statements are fetched at analysis time only and are cached in-process
(module-level dict) for the life of the run — never written to disk or
committed. This repo is public; republishing LeetCode's problem text
into it would not be appropriate. If a slug can't be resolved, the
network call fails, or the problem is LeetCode-premium (content is
null), `get_statement` returns `None` for the text and analysis proceeds
without a statement, same as before this module existed.
"""

import html
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
REQUEST_TIMEOUT = 15

# Browser-ish UA: LeetCode's GraphQL endpoint has been observed to behave
# differently (or not respond usefully) for obviously non-browser clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

QUERY = "query ($slug: String!) { question(titleSlug: $slug) { title content } }"

# NeetCode slug -> LeetCode titleSlug, for the cases where they diverge.
# Anything not listed here is tried as-is (many NeetCode slugs already
# match LeetCode's titleSlug, e.g. "longest-consecutive-sequence",
# "valid-sudoku").
#
# Verified 2026-08-13 by an actual GraphQL fetch against each entry
# (content non-null + title sanity-checked), content sizes at the time:
#   two-integer-sum              -> two-sum                                  (1442 chars)
#   two-integer-sum-ii           -> two-sum-ii-input-array-is-sorted         (2244 chars)
#   duplicate-integer            -> contains-duplicate                       (1308 chars)
#   is-anagram                   -> valid-anagram                            (1121 chars)
#   anagram-groups                -> group-anagrams                          (1844 chars)
#   is-palindrome                 -> valid-palindrome                        (1415 chars)
#   top-k-elements-in-list        -> top-k-frequent-elements                 (1522 chars)
#   products-of-array-discluding-self -> product-of-array-except-self        (1365 chars)
#   string-encode-and-decode      -> encode-and-decode-strings               (LC premium, content is null)
#   buy-and-sell-crypto           -> best-time-to-buy-and-sell-stock         (1249 chars)
#   validate-parentheses          -> valid-parentheses                       (1961 chars)
# Also verified as needing no override (fallback-as-is works):
#   longest-consecutive-sequence  -> longest-consecutive-sequence            (985 chars)
#   valid-sudoku                  -> valid-sudoku                            (3973 chars)
OVERRIDES = {
    "two-integer-sum": "two-sum",
    "two-integer-sum-ii": "two-sum-ii-input-array-is-sorted",
    "duplicate-integer": "contains-duplicate",
    "is-anagram": "valid-anagram",
    "anagram-groups": "group-anagrams",
    "is-palindrome": "valid-palindrome",
    "top-k-elements-in-list": "top-k-frequent-elements",
    "products-of-array-discluding-self": "product-of-array-except-self",
    "string-encode-and-decode": "encode-and-decode-strings",
    "buy-and-sell-crypto": "best-time-to-buy-and-sell-stock",
    "validate-parentheses": "valid-parentheses",
}

# In-process only — see module docstring. Keyed by NeetCode slug, value is
# the resolved statement text or None.
_CACHE: dict[str, str | None] = {}


def resolve_title_slug(neetcode_slug: str) -> str:
    """Map a NeetCode problem slug to its LeetCode titleSlug."""
    return OVERRIDES.get(neetcode_slug, neetcode_slug)


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML->text conversion: strip tags, keep block structure as
    newlines so multi-paragraph sections (like Constraints) stay readable
    and intact rather than running together."""

    # Paragraph-level tags get a blank line after them; list items just a
    # single line break (with a "- " lead-in on the way in), so a bullet
    # list reads like a list rather than one blank-line-separated blob.
    _PARAGRAPH_END_TAGS = {"p", "div", "pre", "h1", "h2", "h3", "h4", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")

    def handle_endtag(self, tag):
        if tag in self._PARAGRAPH_END_TAGS:
            self._parts.append("\n\n")
        elif tag == "li":
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def html_to_text(html_content: str) -> str:
    """Convert LeetCode's `content` HTML field into readable plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html_content)
    parser.close()
    text = html.unescape(parser.get_text())
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_question(title_slug: str) -> dict | None:
    """POST the GraphQL query for `title_slug`. Returns the `question`
    object (may have `content: None` for premium problems), or None if
    the slug doesn't resolve to a question or the request fails."""
    payload = json.dumps({"query": QUERY, "variables": {"slug": title_slug}}).encode("utf-8")
    request = urllib.request.Request(
        LEETCODE_GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None
    return body.get("data", {}).get("question")


def get_statement(neetcode_slug: str) -> tuple[str, str | None]:
    """Resolve `neetcode_slug` and fetch its LeetCode problem statement as
    plain text. Returns `(title_slug, text_or_None)` — `text` is None on
    any resolution failure, network error, or premium (content-null)
    problem, in which case the caller should proceed without a statement.
    Fetched statements are cached in-process only (never written to disk)."""
    title_slug = resolve_title_slug(neetcode_slug)
    if neetcode_slug in _CACHE:
        return title_slug, _CACHE[neetcode_slug]

    text = None
    question = fetch_question(title_slug)
    if question is not None:
        content = question.get("content")
        if content:
            text = html_to_text(content)

    _CACHE[neetcode_slug] = text
    return title_slug, text
