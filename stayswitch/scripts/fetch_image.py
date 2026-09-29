"""Resumable, parallel image fetch from a registry into the local Docker engine.

From this network GHCR serves ~400 KB/s per connection and drops long
transfers; `docker pull` and `crane pull` restart from zero when that happens.
This fetcher downloads blobs over HTTP/1.1 with Range resume and retries,
many blobs at once, into a content-addressed cache shared across images
(same-repo SWE-bench images share their base layers), then assembles a
docker-archive and `docker load`s it.

Usage: python scripts/fetch_image.py IMAGE[=LOCAL_TAG] ... [-j 8]
IMAGE is registry/repo:tag (GHCR anonymous pulls only).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CACHE = Path.home() / ".cache" / "stayswitch" / "blobs"
MANIFEST_TYPES = "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
CHUNK = 1 << 20


class Registry:
    def __init__(self, host: str, repo: str) -> None:
        self.host, self.repo = host, repo
        self._token: str | None = None
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        url = f"https://{self.host}/token?scope=repository:{self.repo}:pull"
        self._token = json.load(urllib.request.urlopen(url, timeout=30))["token"]

    def open(self, path: str, headers: dict[str, str] | None = None, timeout: float = 60):
        with self._lock:
            if self._token is None:
                self._refresh()
        for attempt in range(2):
            req = urllib.request.Request(
                f"https://{self.host}/v2/{self.repo}/{path}",
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": "stayswitch-fetch/0.1", **(headers or {})},
            )
            try:
                return urllib.request.urlopen(req, timeout=timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0:  # token expired
                    with self._lock:
                        self._refresh()
                    continue
                raise
        raise RuntimeError("unreachable")


def parse_ref(ref: str) -> tuple[str, str, str]:
    host, rest = ref.split("/", 1)
    repo, _, tag = rest.rpartition(":") if ":" in rest.split("/")[-1] else (rest, "", "latest")
    return host, repo, tag or "latest"


def fetch_blob(reg: Registry, digest: str, size: int, retries: int = 30) -> Path:
    dest = CACHE / digest.replace(":", "_")
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_suffix(".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        try:
            with reg.open(f"blobs/{digest}", {"Range": f"bytes={have}-"}, timeout=120) as resp, open(part, "ab") as out:
                if resp.status == 200 and have:  # server ignored Range: start over
                    out.truncate(0)
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
        except Exception:
            time.sleep(min(2 * (attempt + 1), 20))
    h = hashlib.sha256()
    with open(part, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    if "sha256:" + h.hexdigest() != digest:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{digest}: checksum mismatch after download")
    part.replace(dest)
    return dest


def get_manifest(ref: str) -> tuple[Registry, dict]:
    host, repo, tag = parse_ref(ref)
    reg = Registry(host, repo)
    with reg.open(f"manifests/{tag}", {"Accept": MANIFEST_TYPES}) as resp:
        return reg, json.load(resp)


def load_image(manifest: dict, local_tag: str) -> None:
    """Assemble a docker-archive from cached blobs and load it under local_tag."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "image.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(CACHE / manifest["config"]["digest"].replace(":", "_"), arcname="config.json")
            layer_names = []
            for i, layer in enumerate(manifest["layers"]):
                name = f"layer{i}.tar.gz"
                tar.add(CACHE / layer["digest"].replace(":", "_"), arcname=name)
                layer_names.append(name)
            index = json.dumps([{"Config": "config.json", "RepoTags": [local_tag], "Layers": layer_names}]).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(index)
            tar.addfile(info, io.BytesIO(index))
        subprocess.run(["docker", "load", "-i", str(archive)], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="REF or REF=LOCAL_TAG")
    ap.add_argument("-j", type=int, default=8, help="parallel blob downloads")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    targets = []  # (ref, local_tag, registry, manifest)
    for spec in args.images:
        ref, _, local = spec.partition("=")
        try:
            reg, manifest = get_manifest(ref)
            targets.append((ref, local or ref, reg, manifest))
        except Exception as exc:
            print(f"{ref}: FAILED manifest {exc}", flush=True)

    # Every blob of every image goes into one pool, deduplicated by digest.
    blobs: dict[str, tuple[Registry, int]] = {}
    for _, _, reg, manifest in targets:
        for b in [manifest["config"], *manifest["layers"]]:
            blobs.setdefault(b["digest"], (reg, b["size"]))
    total = sum(size for _, size in blobs.values())
    print(f"{len(targets)} images, {len(blobs)} unique blobs, {total / 1e9:.2f} GB", flush=True)
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(args.j) as pool:
        futures = {pool.submit(fetch_blob, reg, digest, size): digest for digest, (reg, size) in blobs.items()}
        done = 0
        for fut in as_completed(futures):
            done += blobs[futures[fut]][1]
            try:
                fut.result()
            except Exception as exc:
                errors[futures[fut]] = str(exc)
            print(f"  blobs {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)

    failed = 0
    for ref, local, _, manifest in targets:
        missing = [b["digest"] for b in [manifest["config"], *manifest["layers"]] if b["digest"] in errors]
        if missing:
            failed += 1
            print(f"{ref}: FAILED {len(missing)} blobs", flush=True)
            continue
        load_image(manifest, local)
        print(f"{ref} -> {local}", flush=True)
    sys.exit(1 if failed or len(targets) < len(args.images) else 0)


if __name__ == "__main__":
    main()
