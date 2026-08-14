#!/usr/bin/env python3
"""Announce newly-analyzed submissions to Discord.

Reads the list of this run's newly-analyzed submissions written by
`scripts/analyze.py` (see `analyze.new_this_run_path`), and for each one
posts a message to a Discord channel webhook: a spoilered,
syntax-highlighted screenshot of the source plus a compact stats embed
(problem name/difficulty, time/space complexity, code length) linking back
to https://apriiori.com/projects/neetcode/.

Degrades gracefully and silently:
  - `DISCORD_WEBHOOK_URL` unset -> exit 0, no output (the webhook doesn't
    exist yet; April is setting it up).
  - Nothing new this run (list missing or empty) -> exit 0, no output.
  - Rendering the source screenshot fails (freeze missing/broken) -> the
    message still posts, just without the image.

The screenshot is rendered with charmbracelet's `freeze`
(https://github.com/charmbracelet/freeze), a single released binary — see
FREEZE_BIN below for how the script finds it.

Usage:
    python3 scripts/announce.py             # posts for real (needs webhook)
    python3 scripts/announce.py --dry-run    # prints payloads, posts nothing
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze
import statements

REPO_ROOT = Path(__file__).resolve().parent.parent

# A single Go-released binary; no third-party Python deps needed to run it.
# In CI this is downloaded and pinned to a specific version (see
# .github/workflows/analyze.yml); locally, FREEZE_BIN lets you point at
# wherever you put the binary (e.g. after downloading a release yourself),
# defaulting to whatever's on PATH.
FREEZE_BIN = os.environ.get("FREEZE_BIN", "freeze")
FREEZE_THEME = "dracula"
FREEZE_LANGUAGE = "python"

# freeze writes an SVG and then rasterizes it, preferring `rsvg-convert` when
# it's on PATH and otherwise falling back to resvg compiled to WebAssembly and
# run under wazero (freeze's png.go). That fallback is the flaky one: the wasm
# module runs on its own stack, and Go's GC has crashed mid-render trying to
# walk it — "SIGSEGV in runtime.scanstack" and "split stack overflow" are both
# that. Turning the GC off keeps it from scanning anything during these
# sub-second renders. The real fix is having rsvg-convert installed (CI does,
# via .github/actions/setup-render), which skips the wasm path entirely; this
# is just insurance for machines that don't.
FREEZE_ENV = {**os.environ, "GOGC": "off"}

PROJECT_URL = "https://apriiori.com/projects/neetcode/"

# Discord's own semantic colors (from their branding guide), matching
# NeetCode's Easy/Medium/Hard labels.
DIFFICULTY_COLORS = {
    "Easy": 0x57F287,
    "Medium": 0xFEE75C,
    "Hard": 0xED4245,
}
DEFAULT_COLOR = 0x5865F2  # Discord blurple, for an unknown/missing difficulty

SLEEP_BETWEEN_POSTS = 2.0


def humanize_slug(slug: str) -> str:
    return slug.replace("-", " ").title()


def difficulty_color(difficulty: str | None) -> int:
    return DIFFICULTY_COLORS.get(difficulty, DEFAULT_COLOR)


def fetch_problem_meta(slug: str) -> dict:
    """Best-effort problem name/difficulty for the embed title, via
    statements.fetch_problem — same NeetCode metadata endpoint analyze.py's
    statement fetch uses, which already degrades to None on any failure.
    Falls back to a humanized slug and no difficulty."""
    data = None
    try:
        data = statements.fetch_problem(slug)
    except Exception:
        data = None
    name = None
    difficulty = None
    if isinstance(data, dict):
        name = data.get("name") or None
        difficulty = data.get("difficulty") or None
    return {"name": name or humanize_slug(slug), "difficulty": difficulty}


def spoiler_filename(slug: str, number: int) -> str:
    return f"SPOILER_{slug}-submission-{number}.png"


def stats_header(analysis: dict, meta: dict) -> str:
    """Comment block prepended to the source before rendering, so the stats
    live inside the (spoilered) image itself."""
    complexity = analysis.get("complexity") or {}
    golf = analysis.get("golf") or {}
    name = meta.get("name") or "Untitled problem"
    difficulty = meta.get("difficulty")
    title = f"{name} ({difficulty})" if difficulty else name
    time_average = complexity.get("time_average", "?")
    time_worst = complexity.get("time_worst", "?")
    space = complexity.get("space", "?")
    tokens = golf.get("tokens", "?")
    lines = golf.get("non_blank_lines", "?")
    chars = golf.get("characters", "?")
    return (
        f"# {title}\n"
        f"# time: {time_average} avg · {time_worst} worst | space: {space}\n"
        f"# length: {tokens} tokens · {lines} lines · {chars} chars\n"
        "\n"
    )


def render_source_image(source_path: Path, header: str = "") -> bytes | None:
    """Render `source_path` (with `header` prepended) to a PNG via freeze.
    Returns the PNG bytes, or None on any failure (freeze missing, non-zero
    exit, timeout, or no output produced) — the caller falls back to posting
    without an image. Retries once, since the crash described in
    FREEZE_ENV can hit one invocation and not the next."""
    for attempt in range(2):
        result = _render_once(source_path, header)
        if result is not None:
            return result
    return None


def _render_once(source_path: Path, header: str) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        annotated_path = Path(tmp_dir) / source_path.name
        annotated_path.write_text(
            header + source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        out_path = Path(tmp_dir) / "out.png"
        cmd = [
            FREEZE_BIN,
            str(annotated_path),
            "-o",
            str(out_path),
            "-l",
            FREEZE_LANGUAGE,
            "-t",
            FREEZE_THEME,
            "--window",
        ]
        try:
            # stdin must be detached: freeze silently prefers a piped stdin
            # over the file argument, and CI steps run with an empty pipe on
            # stdin — which reads as "No input" and kills the render.
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                env=FREEZE_ENV,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", "replace")[:500]
            stdout = (exc.stdout or b"").decode("utf-8", "replace")[:500]
            print(
                f"[announce] freeze exited {exc.returncode}: stderr={stderr!r} stdout={stdout!r}",
                file=sys.stderr,
            )
            return None
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            print(f"[announce] freeze failed to run: {exc}", file=sys.stderr)
            return None
        if not out_path.is_file():
            print("[announce] freeze produced no output file", file=sys.stderr)
            return None
        return out_path.read_bytes()


# The image is deliberately NOT referenced as embed["image"]: a SPOILER_
# attachment rendered inside an embed loses its spoiler blur (discord-api-docs
# issue #1235). Left as a plain attachment, Discord shows it below the embed,
# blurred until clicked.
def build_embed(meta: dict, slug: str) -> dict:
    # Only spoiler-safe facts here: the problem's name/difficulty and the
    # progress-page link. Complexity and length stats can hint at the
    # approach, so they live inside the SPOILER_ image (stats_header), not
    # in the visible embed.
    name = meta.get("name") or "Untitled problem"
    difficulty = meta.get("difficulty")
    title = f"{name} ({difficulty})" if difficulty else name

    return {
        "title": title,
        "url": f"{PROJECT_URL}#{slug}",
        "color": difficulty_color(difficulty),
        "description": f"[problem on NeetCode](https://neetcode.io/problems/{slug})",
    }


def build_announcement(record: dict) -> tuple[dict, bytes | None, str]:
    """Build (payload, image_bytes_or_None, filename) for one new-this-run
    record from analyze.py's list."""
    analysis_path = REPO_ROOT / record["analysis_file"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    source_path = REPO_ROOT / record["file"]

    slug = record["slug"]
    number = record["number"]
    filename = spoiler_filename(slug, number)

    meta = fetch_problem_meta(slug)
    image_bytes = render_source_image(source_path, stats_header(analysis, meta))

    embed = build_embed(meta, slug)
    payload = {"embeds": [embed]}
    return payload, image_bytes, filename


def encode_multipart(payload: dict, filename: str | None, image_bytes: bytes | None) -> tuple[bytes, str]:
    """Encode a Discord webhook multipart/form-data body: a `payload_json`
    field plus, when an image is available, a `files[0]` field carrying it.
    Returns (body_bytes, content_type_header_value)."""
    boundary = uuid.uuid4().hex
    parts = []

    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="payload_json"\r\n'
            "Content-Type: application/json\r\n\r\n"
            f"{json.dumps(payload)}\r\n"
        ).encode("utf-8")
    )

    if filename and image_bytes is not None:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
        parts.append(header + image_bytes + b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def post_webhook(webhook_url: str, payload: dict, image_bytes: bytes | None, filename: str) -> bool:
    """POST one announcement to the Discord webhook. Returns True on
    success; on failure, logs to stderr and returns False rather than
    raising, so one bad post doesn't stop the rest of the run."""
    body, content_type = encode_multipart(payload, filename if image_bytes else None, image_bytes)
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": content_type,
            # Discord's edge 403s Python's default urllib User-Agent
            "User-Agent": "neetcode-submissions announcer (github.com/ApriiSR/neetcode-submissions)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        detail = ""
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = f" — {exc.read().decode('utf-8', 'replace')[:300]}"
            except OSError:
                pass
        print(f"[announce] failed to post {filename}: {exc}{detail}", file=sys.stderr)
        return False


def load_new_records() -> list:
    path = analyze.new_this_run_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    # Re-analyses (version bumps, error retries) regenerate records for old
    # solves — announcing those would flood the channel with stale problems.
    return [r for r in data if not r.get("reanalysis")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print payloads instead of posting")
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not args.dry_run and not webhook_url:
        # The webhook doesn't exist yet — quiet, expected no-op.
        return 0

    new_records = load_new_records()
    if not new_records:
        return 0

    had_error = False
    for i, record in enumerate(new_records):
        payload, image_bytes, filename = build_announcement(record)

        if args.dry_run:
            print(json.dumps(payload, indent=2))
            if image_bytes:
                print(f"[dry-run] would attach {filename} ({len(image_bytes)} bytes)")
            else:
                print(f"[dry-run] no image ({filename} not rendered)")
        else:
            ok = post_webhook(webhook_url, payload, image_bytes, filename)
            had_error = had_error or not ok
            if ok and image_bytes is None:
                # The image is the announcement — an embed without it is a
                # title and a link. Posting anyway beats silence, but this
                # should turn the run red rather than hide in stderr, since
                # the only other place it shows up is Discord.
                print(f"::warning::posted {filename} without its screenshot (render failed)")
                had_error = True

        if i < len(new_records) - 1:
            time.sleep(SLEEP_BETWEEN_POSTS)

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
