"""Select SWE-bench Verified tasks and fetch arm64 images for them.

The Harbor tasks build FROM ``swebench/sweb.eval.x86_64.<id>`` on Docker Hub,
which is blocked here and x86-only. Epoch AI publishes arm64 builds of the same
environments on GHCR (``ghcr.io/epoch-research/swe-bench.eval.arm64.<instance>``);
this script pulls those and tags them under the name each Dockerfile expects,
so Harbor's build finds the base image locally.

GHCR runs at ~400 KB/s from here, so selection prefers smaller images (the
largest quartile is excluded) and pulls run in the background.

  python scripts/swe_images.py select TASKS_DIR [--n-per-bucket 10] [--seed 0]  -> configs/swe_tasks.txt
  python scripts/swe_images.py pull TASKS_DIR [-j 8]                             -> fetches + tags listed tasks
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tomllib
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_LIST = ROOT / "configs" / "swe_tasks.txt"
EPOCH = "ghcr.io/epoch-research/swe-bench.eval.arm64.{instance}:latest"
BUCKETS = ("<15 min fix", "15 min - 1 hour")


def epoch_repo(instance: str) -> str:
    return f"epoch-research/swe-bench.eval.arm64.{instance}"


def image_size(instance: str) -> int | None:
    repo = epoch_repo(instance)
    try:
        tok = json.load(urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{repo}:pull", timeout=30))["token"]
        req = urllib.request.Request(
            f"https://ghcr.io/v2/{repo}/manifests/latest",
            headers={
                "Authorization": f"Bearer {tok}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json",
            },
        )
        manifest = json.load(urllib.request.urlopen(req, timeout=30))
        return sum(layer["size"] for layer in manifest.get("layers", []))
    except Exception:
        return None  # not published for arm64, or unreachable


def load_tasks(tasks_dir: Path) -> list[dict]:
    tasks = []
    for toml_path in sorted(tasks_dir.glob("*/task.toml")):
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        instance = toml_path.parent.name
        dockerfile = (toml_path.parent / "environment" / "Dockerfile").read_text()
        base = re.search(r"^FROM\s+(\S+)", dockerfile, re.M).group(1)
        tasks.append(
            {
                "instance": instance,
                "repo": instance.split("__")[0],
                "difficulty": cfg.get("metadata", {}).get("difficulty"),
                "base": base,
            }
        )
    return tasks


def select(tasks_dir: Path, n_per_bucket: int, seed: int, max_per_repo: int = 3) -> None:
    rng = random.Random(seed)
    tasks = [t for t in load_tasks(tasks_dir) if t["difficulty"] in BUCKETS]
    rng.shuffle(tasks)
    # Size only a shuffled prefix: enough candidates to fill both buckets after filtering.
    pool = tasks[: n_per_bucket * 16]
    with ThreadPoolExecutor(8) as ex:
        for t, size in zip(pool, ex.map(lambda t: image_size(t["instance"]), pool)):
            t["size"] = size
    sized = [t for t in pool if t.get("size")]
    cutoff = sorted(t["size"] for t in sized)[int(0.75 * len(sized))]
    chosen, per_repo = [], Counter()
    for bucket in BUCKETS:
        n = 0
        for t in sized:
            if t["difficulty"] == bucket and t["size"] <= cutoff and per_repo[t["repo"]] < max_per_repo and n < n_per_bucket:
                chosen.append(t)
                per_repo[t["repo"]] += 1
                n += 1
    lines = [
        f"# SWE-bench Verified pilot: {n_per_bucket} per difficulty bucket, seed {seed}, <= {max_per_repo} per repo,",
        f"# arm64 image <= {cutoff / 1e9:.2f} GB (largest quartile of sampled candidates excluded).",
        "# instance  difficulty  image_GB",
    ]
    lines += [f"{t['instance']}  # {t['difficulty']}  {t['size'] / 1e9:.2f}" for t in chosen]
    TASK_LIST.write_text("\n".join(lines) + "\n")
    total = sum(t["size"] for t in chosen)
    print(f"selected {len(chosen)} tasks, {total / 1e9:.1f} GB to pull -> {TASK_LIST.relative_to(ROOT)}")
    print(dict(Counter(t["repo"] for t in chosen)))


def listed_instances() -> list[str]:
    return [line.split("#")[0].strip() for line in TASK_LIST.read_text().splitlines() if line.split("#")[0].strip()]


def pull(tasks_dir: Path, jobs: int) -> None:
    """Fetch on the host with scripts/fetch_image.py (resumable, parallel); `docker pull` in the VM is ~8x slower."""
    wanted = set(listed_instances())
    tasks = [t for t in load_tasks(tasks_dir) if t["instance"] in wanted]
    missing = [t for t in tasks if subprocess.run(["docker", "image", "inspect", t["base"]], capture_output=True).returncode]
    print(f"{len(tasks) - len(missing)} of {len(tasks)} base images already present", flush=True)
    if not missing:
        return
    specs = [f"{EPOCH.format(instance=t['instance'])}={t['base']}" for t in missing]
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_image.py"), *specs, "-j", str(jobs)]).returncode
    sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "pull"])
    ap.add_argument("tasks_dir", type=Path, help="exported swe-bench-verified task directory")
    ap.add_argument("--n-per-bucket", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-j", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "select":
        select(args.tasks_dir, args.n_per_bucket, args.seed)
    else:
        pull(args.tasks_dir, args.j)


if __name__ == "__main__":
    main()
