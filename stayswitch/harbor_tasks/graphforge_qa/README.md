# graphforge_qa: OfficeQA-style table-QA tasks for Harbor

129 tasks derived from `groundhogLLM/GraphForge-SFT-2169`. That dataset is trajectory-only: the real public workspace
files are not bundled. We recovered the data tables from trajectories where the teacher agent `cat`-ed a whole
CSV/JSON (full output, not truncated), 155 task workspaces / 219 files, and wrote new questions over them.
The original GraphForge prompts and rubrics (docx/xlsx deliverables, LLM-judged) are not used.

Each task: a multi-step question over 1-3 real tables (FDIC bank financials, Treasury yields, EPA, FDOT, NSF, ...),
a single answer (number / string / list) written to `/app/answer.txt`.

## Layout (Harbor task format)
    tasks/<id>/instruction.md  task.toml  environment/{Dockerfile,data/}  tests/{test.sh,verify.py,expected.json}  solution/{solve.sh,solve.py}

## Deterministic verifier
`tests/verify.py` (stdlib only, no LLM/network) compares `/app/answer.txt` to `tests/expected.json` and writes `/logs/verifier/reward.txt`
(1/0). number: `math.isclose` with per-task rel/abs tolerance (strips `,` `$` `%`); string: case/punctuation-normalised exact;
list: order-insensitive. `solution/solve.py` recomputes the answer from `/app/data`, so each task is an oracle-checked
(solve -> verify = 1, wrong answer -> 0) program, not a stored guess.

## Rebuild
`authoring/*.json` are the specs; `python3 build.py [spec...]` regenerates `tasks/` and refuses any task whose solver does not
reproduce the declared answer. `GF_POOL` points at the recovered data pool (not committed; the data is already copied into each task).

## Caveats
- Source data licensing: public datasets, but the dataset card asserts no blanket license; check before redistributing.
- Questions were LLM-authored and oracle-checked, not human-reviewed. Spot-check ambiguity before using as a benchmark.
- FDIC NETINC columns are year-to-date; questions were worded accordingly.
- Not yet run through Harbor/docker with a real agent (no docker here); verifier + oracle were exercised locally.
