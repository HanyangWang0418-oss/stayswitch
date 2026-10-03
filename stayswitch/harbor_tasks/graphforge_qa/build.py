"""Build Harbor task dirs from authoring/*.json specs.

spec: {id, source_task_id, files:[pool filenames], question, answer_type, answer, rel_tol?, abs_tol?, numeric?, solve_py}
solve_py reads the data from DATA_DIR (env, default /app/data) and prints the answer on stdout.
Build refuses a task unless solve_py (run on the real files) reproduces the declared answer through verify.py.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
POOL = Path(os.environ.get("GF_POOL", "/tmp/claude-0/-home-user-stayswitch/66dda051-9418-549c-9307-86ea7c035295/scratchpad/gf/pool"))

TOML = '''version = "1.0"

[metadata]
author_name = "graphforge-derived"
difficulty = "{difficulty}"
category = "table-qa"
tags = ["officeqa-style", "deterministic-verifier", "source:GraphForge-SFT-2169"]
source_task_id = "{src}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 2048
storage_mb = 2048
'''
DOCKER = '''FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.2.3 openpyxl==3.1.5
WORKDIR /app
COPY data/ /app/data/
'''
INSTR = '''{question}

The source files are in `/app/data/`. Work out the answer from them, then write ONLY the final answer
{fmt} to `/app/answer.txt`.
'''
FMT = {"number": "(a single number, no units or commentary; plain digits, e.g. 1234.56)",
       "string": "(a single short string, exactly as it appears in the data)",
       "list": "(one item per line)"}
TEST_SH = '''#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify.py /tests/expected.json /app/answer.txt /logs/verifier/reward.txt
'''
SOLVE_SH = '''#!/bin/bash
python3 /solution/solve.py > /app/answer.txt
'''

def build(spec_path):
    s = json.loads(spec_path.read_text())
    d = HERE / "tasks" / s["id"]
    if d.exists(): shutil.rmtree(d)
    for sub in ("environment/data", "tests", "solution"): (d / sub).mkdir(parents=True)
    for f in s["files"]: shutil.copy(POOL / s["source_task_id"] / f, d / "environment/data" / f)
    exp = {"type": s["answer_type"], "value": s["answer"]}
    for k in ("rel_tol", "abs_tol", "numeric"):
        if k in s: exp[k] = s[k]
    (d / "tests/expected.json").write_text(json.dumps(exp, indent=1))
    shutil.copy(HERE / "verify.py", d / "tests/verify.py")
    (d / "tests/test.sh").write_text(TEST_SH); (d / "solution/solve.sh").write_text(SOLVE_SH)
    (d / "solution/solve.py").write_text(s["solve_py"])
    (d / "environment/Dockerfile").write_text(DOCKER)
    (d / "instruction.md").write_text(INSTR.format(question=s["question"].strip(), fmt=FMT[s["answer_type"]]))
    (d / "task.toml").write_text(TOML.format(difficulty=s.get("difficulty", "medium"), src=s["source_task_id"]))
    for sh in (d / "tests/test.sh", d / "solution/solve.sh"): sh.chmod(0o755)
    # oracle check: run solve.py against the shipped data, score with the shipped verifier
    env = dict(os.environ, DATA_DIR=str(d / "environment/data"))
    r = subprocess.run([sys.executable, str(d / "solution/solve.py")], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode: raise SystemExit(f"{s['id']}: solve.py failed\n{r.stderr}")
    with tempfile.TemporaryDirectory() as t:
        ans = Path(t) / "answer.txt"; ans.write_text(r.stdout)
        rew = Path(t) / "r.txt"
        subprocess.run([sys.executable, str(d / "tests/verify.py"), str(d / "tests/expected.json"), str(ans), str(rew)], check=True, capture_output=True)
        if rew.read_text() != "1":
            shutil.rmtree(d); raise SystemExit(f"{s['id']}: oracle output {r.stdout!r} != declared answer {s['answer']!r}")
        # negative control: empty / wrong answer must score 0
        ans.write_text("zzz-wrong"); subprocess.run([sys.executable, str(d / "tests/verify.py"), str(d / "tests/expected.json"), str(ans), str(rew)], check=True, capture_output=True)
        assert rew.read_text() == "0"
    print("ok", s["id"])

if __name__ == "__main__":
    for p in sorted((HERE / "authoring").glob("*.json")) if len(sys.argv) < 2 else map(Path, sys.argv[1:]):
        try: build(p)
        except SystemExit as e: print("FAIL", e)
