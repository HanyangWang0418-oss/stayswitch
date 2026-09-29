"""Build Harbor task images locally under the tag each task would pull.

Terminal-Bench 2 tasks name prebuilt images on Docker Hub (``alexgshaw/<task>:<date>``);
from this network Docker Hub is blocked and the mirrors refuse non-official
repositories. Every task ships the Dockerfile the image was built from, on an
official base image the mirrors do serve, so building it here under the same
tag lets Harbor find the image locally and skip the pull. Both baseline jobs
then share one image per task.

Deviation from the published images: apt sources inside the image point at a
nearby mirror (same packages; upstream hosts are slow from here), and any
unpinned package may resolve to a newer version than on the build date.

Usage: python scripts/prebuild_images.py [dataset_dir] [--only name ...] [-j N]
Default dataset_dir: ~/.cache/harbor/tasks/packages/terminal-bench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stayswitch.agents import DEFAULT_APT_MIRROR, apt_mirror_command


def task_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/*/task.toml"))


def image_tag(task: Path) -> str | None:
    with open(task / "task.toml", "rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("environment", {}).get("docker_image")


def exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def mirrored_dockerfile(dockerfile: Path, mirror: str) -> str:
    """The task's Dockerfile with apt pointed at ``mirror`` right after each FROM (a no-op on non-apt images)."""
    lines = []
    for line in dockerfile.read_text().splitlines():
        lines.append(line)
        if line.strip().upper().startswith("FROM "):
            lines.append(f"RUN {apt_mirror_command(mirror)}")
    return "\n".join(lines) + "\n"


def build(task: Path, log_dir: Path, mirror: str, retries: int) -> tuple[str, str]:
    name = task.parent.name
    tag = image_tag(task)
    if not tag:
        return name, "no docker_image (Harbor builds it itself)"
    if exists(tag):
        return name, f"exists {tag}"
    log = log_dir / f"{name}.log"
    context = task / "environment"
    cmd = ["docker", "build", "-t", tag]
    if mirror:
        dockerfile = log_dir / f"{name}.Dockerfile"
        dockerfile.write_text(mirrored_dockerfile(context / "Dockerfile", mirror))
        cmd += ["-f", str(dockerfile)]
    rc = 1
    with open(log, "w") as out:
        # The registry mirror intermittently rejects base-image metadata lookups; a retry usually passes.
        for attempt in range(retries):
            out.write(f"### attempt {attempt + 1}\n")
            out.flush()
            rc = subprocess.run(cmd + [str(context)], stdout=out, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                break
    return name, f"built {tag}" if rc == 0 else f"FAILED (see {log})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(Path.home() / ".cache/harbor/tasks/packages/terminal-bench"))
    ap.add_argument("--only", nargs="*", help="task names to build")
    ap.add_argument("-j", type=int, default=2, help="parallel builds")
    ap.add_argument("--apt-mirror", default=DEFAULT_APT_MIRROR, help="apt mirror baked into images; '' to keep upstream")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    tasks = task_dirs(Path(args.root))
    if args.only:
        tasks = [t for t in tasks if t.parent.name in set(args.only)]
    log_dir = Path(__file__).resolve().parent.parent / "runs" / "prebuild"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(args.j) as pool:
        for name, status in pool.map(lambda t: build(t, log_dir, args.apt_mirror, args.retries), tasks):
            print(f"{name:32s} {status}", flush=True)
            failed += status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
