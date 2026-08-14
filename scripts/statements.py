"""Fetch NeetCode problem statements for use in complexity analysis.

`scripts/analyze.py` asks K3 to review a solution's complexity, but the
solution's source code alone doesn't carry the problem's stated
constraints (e.g. "1 <= prices.length <= 100") — so K3 sometimes flags a
correct reliance on those constraints as an "unjustified assumption".
Fetching the actual problem text and including it in the prompt fixes
that.

An earlier version of this module sourced statements from LeetCode's
GraphQL endpoint via a NeetCode-slug -> LeetCode-titleSlug mapping, on
the assumption that NeetCode itself had no usable public source. That
assumption turned out to be wrong in a way that mattered: NeetCode's own
constraints can *differ* from LeetCode's (e.g. `buy-and-sell-crypto`
caps `prices[i]` at 100 on NeetCode vs. 10^4 on LeetCode), so a
LeetCode-sourced statement could itself introduce the same
statement-vs-code mismatch this module exists to prevent — K3 flagged
correct NeetCode-constrained code as buggy against LeetCode's spec.

The actual NeetCode data source was found 2026-08-13 by watching the
neetcode.io SPA's network traffic while browsing a problem page: it
POSTs to `getProblemMetadataFunctionHttp` and gets back the full problem
JSON, works anonymously, and needs no slug mapping at all — the
"problemId" it wants is exactly the NeetCode slug, which is also this
repo's directory name for that problem. This endpoint is undocumented
and internal to the SPA, so it could change shape or go away without
notice; see `get_statement`'s degradation behavior for what happens then.

Statements are fetched at analysis time only and are cached in-process
(module-level dict) for the life of the run — never written to disk or
committed. This repo is public; republishing NeetCode's problem text
into it would not be appropriate. If the slug isn't found, the network
call fails, or the response doesn't have the shape expected,
`get_statement` returns `None` for the text and analysis proceeds
without a statement.
"""

import json
import re
import urllib.error
import urllib.request

NEETCODE_METADATA_URL = "https://neetcode.io/api/getProblemMetadataFunctionHttp"
REQUEST_TIMEOUT = 15

# Browser-ish UA: matches what was captured from the SPA; unclear whether
# the endpoint actually requires it, but there's no reason to find out.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# The description's markdown runs: statement, examples, a "**Constraints:**"
# list, then a series of `<details class="hint-accordion">` (or similar)
# accordion blocks — Topics, "Recommended Time & Space Complexity", Hint
# 1..N, Company Tags. Everything from the first `<details` onward must NOT
# reach K3: the "Recommended Time & Space Complexity" block in particular
# states the expected answer outright, which would make the complexity
# analysis worthless (and the hints are meant to stay hidden from someone
# solving the problem).
DETAILS_MARKER = "<details"

# In-process only — see module docstring. Keyed by NeetCode slug, value is
# the resolved statement text or None.
_CACHE: dict[str, str | None] = {}


def truncate_before_details(description: str) -> str:
    """Cut everything from the first `<details` accordion block onward —
    hints, the recommended-complexity spoiler, and company tags."""
    idx = description.find(DETAILS_MARKER)
    if idx != -1:
        description = description[:idx]
    return description


def clean_markdown(description: str) -> str:
    """Light cleanup of NeetCode's markdown: drop stray `<br>` tags left
    over from the accordion layout and collapse the resulting run of blank
    lines, but otherwise leave the markdown (backticks, **bold**, code
    fences, the `**Constraints:**` list) exactly as NeetCode wrote it."""
    description = re.sub(r"<br\s*/?>", "", description, flags=re.IGNORECASE)
    description = re.sub(r"\n{3,}", "\n\n", description)
    return description.strip()


def fetch_problem(neetcode_slug: str) -> dict | None:
    """POST to NeetCode's problem-metadata endpoint. Returns the problem's
    data dict (with a "description" key, among others), or None if the
    slug isn't found or the request fails."""
    payload = json.dumps({"data": {"problemId": neetcode_slug}}).encode("utf-8")
    request = urllib.request.Request(
        NEETCODE_METADATA_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, dict):
        # {"data": null} is the observed shape for an unknown problemId.
        return None
    return data


def get_statement(neetcode_slug: str) -> tuple[str, str | None]:
    """Fetch `neetcode_slug`'s NeetCode problem statement as cleaned
    markdown text, with hints/complexity-spoiler/company-tags stripped.
    Returns `(neetcode_slug, text_or_None)` — `text` is None on any
    not-found, network, or unexpected-shape failure, in which case the
    caller should proceed without a statement. Fetched statements are
    cached in-process only (never written to disk)."""
    if neetcode_slug in _CACHE:
        return neetcode_slug, _CACHE[neetcode_slug]

    text = None
    data = fetch_problem(neetcode_slug)
    if data is not None:
        description = data.get("description")
        if isinstance(description, str) and description.strip():
            text = clean_markdown(truncate_before_details(description))
            if not text:
                text = None

    _CACHE[neetcode_slug] = text
    return neetcode_slug, text
