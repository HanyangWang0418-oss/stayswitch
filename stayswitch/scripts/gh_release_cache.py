"""Caching mirror for GitHub release downloads, for task containers on a flaky network.

Task verifiers install uv with `curl https://astral.sh/uv/<v>/install.sh | sh`,
which fetches the binary from GitHub Releases; from this network that often
fails mid-transfer and the trial scores 0 for reasons unrelated to the agent.
Point the installer here with UV_INSTALLER_GITHUB_BASE_URL=http://host.docker.internal:<port>.
A miss is fetched from github.com with retries and kept on disk.

Usage: python scripts/gh_release_cache.py [port] [cache_dir]
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "https://github.com"
RETRIES = 6


def fetch(path: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(UPSTREAM + path, headers={"User-Agent": "gh-release-cache/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors are the reason this exists
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{path}: {last}")


class Handler(BaseHTTPRequestHandler):
    cache_dir: Path

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if "/releases/download/" not in path or ".." in path:
            self.send_error(404, "only GitHub release assets are mirrored")
            return
        dest = self.cache_dir / path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                fetch(path, dest)
            except RuntimeError as exc:
                self.send_error(502, str(exc))
                return
        self.send_response(200)
        self.send_header("Content-Length", str(dest.stat().st_size))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        with open(dest, "rb") as f:
            shutil.copyfileobj(f, self.wfile)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    Handler.cache_dir = Path(sys.argv[2] if len(sys.argv) > 2 else Path.home() / ".cache/stayswitch/gh-releases")
    Handler.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"mirroring {UPSTREAM} releases on :{port}, cache {Handler.cache_dir}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
