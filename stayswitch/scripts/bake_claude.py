"""Bake Claude Code into the SWE-bench task base images so Harbor skips its per-trial install.

Harbor's claude-code agent runs `command -v claude` (with ~/.local/bin on PATH) before installing; when it
succeeds the apt + bootstrap step, about 5 minutes per trial from here, is skipped. The SWE-bench task
Dockerfiles build FROM ``swebench/sweb.eval.x86_64.<owner>_1776_<repo>-<n>:latest`` (the arm64 images that
scripts/swe_images.py tags under that name), so adding one layer to that tag is enough.

  python scripts/bake_claude.py tools [--version X]   build the shared Claude Code layer once (~5 min)
  python scripts/bake_claude.py bake configs/swe_tasks.txt   add the layer to every listed task's image
  python scripts/bake_claude.py restore configs/swe_tasks.txt   point the tags back at the original images

The layer is /root/.local only (the claude binary), so it does not change what a terminus-2 run sees. The
original image of each task is kept as ``stayswitch-orig/<task>:latest``; ``bake`` is idempotent and always
rebuilds from that backup, and the tag is moved with a single ``docker tag`` so running jobs never see a gap.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_IMAGE = "stayswitch-cc-tools:latest"
LABEL = "stayswitch.claude_code"


def sh(*cmd: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, input=stdin)


def base_tag(task: str) -> str:
    return "swebench/sweb.eval.x86_64." + task.replace("__", "_1776_") + ":latest"


def backup_tag(task: str) -> str:
    return f"stayswitch-orig/{task}:latest"


def exists(tag: str) -> bool:
    return sh("docker", "image", "inspect", tag, check=False).returncode == 0


def tasks_from(path: str) -> list[str]:
    lines = (line.split("#")[0].strip() for line in Path(path).read_text().splitlines())
    return [line.split()[0] for line in lines if line]


def build(dockerfile: str, tag: str) -> None:
    with tempfile.TemporaryDirectory() as ctx:
        out = sh("docker", "build", "-q", "-f", "-", "-t", tag, ctx, stdin=dockerfile)
    print(f"built {tag} {out.stdout.strip()[:19]}", flush=True)


def cmd_tools(version: str, first_task: str) -> None:
    base = backup_tag(first_task) if exists(backup_tag(first_task)) else base_tag(first_task)
    arg = f" -s -- {version}" if version else ""
    build(
        f"FROM {base}\n"
        f"RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash{arg}"
        f" && /root/.local/bin/claude --version\n",
        TOOLS_IMAGE,
    )
    print(sh("docker", "run", "--rm", "--entrypoint", "sh", TOOLS_IMAGE, "-c", "/root/.local/bin/claude --version").stdout.strip())


def cmd_bake(tasks: list[str]) -> None:
    if not exists(TOOLS_IMAGE):
        sys.exit(f"{TOOLS_IMAGE} missing: run `bake_claude.py tools` first")
    version = sh("docker", "run", "--rm", "--entrypoint", "sh", TOOLS_IMAGE, "-c", "/root/.local/bin/claude --version").stdout.split()[0]
    for task in tasks:
        tag = base_tag(task)
        if not exists(backup_tag(task)):
            if not exists(tag):
                print(f"skip {task}: no local image {tag}")
                continue
            sh("docker", "tag", tag, backup_tag(task))
        staging = f"stayswitch-cc-staging/{task}:latest"
        build(
            f"FROM {backup_tag(task)}\nCOPY --from={TOOLS_IMAGE} /root/.local /root/.local\nLABEL {LABEL}={version}\n",
            staging,
        )
        sh("docker", "tag", staging, tag)  # atomic: the old tag keeps serving until this returns
        sh("docker", "rmi", staging)
        print(f"baked {task} (claude {version})", flush=True)


def cmd_restore(tasks: list[str]) -> None:
    for task in tasks:
        if exists(backup_tag(task)):
            sh("docker", "tag", backup_tag(task), base_tag(task))
            print(f"restored {task}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tools")
    t.add_argument("--version", default="", help="Claude Code version to pin (default: latest)")
    t.add_argument("--task", default="psf__requests-5414", help="task whose image supplies glibc/arch for the build")
    for name in ("bake", "restore"):
        sub.add_parser(name).add_argument("task_list")
    args = ap.parse_args()
    if args.cmd == "tools":
        cmd_tools(args.version, args.task)
    elif args.cmd == "bake":
        cmd_bake(tasks_from(args.task_list))
    else:
        cmd_restore(tasks_from(args.task_list))


if __name__ == "__main__":
    main()
