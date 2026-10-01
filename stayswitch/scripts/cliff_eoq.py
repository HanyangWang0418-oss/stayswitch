"""Run the official CliffCompaction proxy with an EOQ compaction threshold.

Everything about *how* history is compacted is CliffCompaction's own code
(github.com/nguyenvuthientrang/cliffcompaction, pinned in pyproject). Only
*when* changes: instead of one fixed ``--threshold``, each request gets

    L* = L0 + sqrt(2 g C / c_r)

where g is the context growth per turn observed in this request's history, L0 the
estimated size right after a compaction (head + summary + kept turns), c_r the
cached-read price and C the cost of one compaction (cache rewrite of L0 plus
``--extra-steps`` redone steps). See stayswitch.context.eoq_threshold.

Usage: python scripts/cliff_eoq.py --model strong --extra-steps 2 -- serve --port 8257 --keep-recent 3 \
           --openai-upstream http://127.0.0.1:4001/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cliffcompaction import cli as cliff_cli
from cliffcompaction import engine as cliff_engine
from cliffcompaction.engine import estimate_tokens

from stayswitch.context import eoq_threshold
from stayswitch.pricing import PriceTable

ROOT = Path(__file__).resolve().parent.parent


def install(model: str, extra_steps: float, log_path: Path, ceiling: int, min_gap: int) -> None:
    price = PriceTable.load(ROOT / "configs" / "prices.toml")[model]
    rule = eoq_threshold(price, extra_steps=extra_steps, ceiling=ceiling)
    original = cliff_engine.Engine.prepare
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare(self, body, dialect):
        msgs = body[dialect.messages_key]
        roles = [m.get("role") for m in msgs]
        first_assistant = roles.index("assistant") if "assistant" in roles else len(msgs)
        turns = max(roles.count("assistant"), 1)
        head = estimate_tokens({**body, dialect.messages_key: msgs[:first_assistant]})
        total = estimate_tokens(body)  # the agent's full, uncompacted history
        growth = max((total - head) / turns, 50.0)
        # Right after a compaction: head + the summary message (~one turn) + kept turns.
        l0 = head + (self.cfg.keep_recent + 1) * growth
        threshold = max(rule({"growth": growth}, total, l0), l0 + min_gap * growth)
        self.cfg.threshold_tokens = int(threshold)  # prepare() is synchronous: no interleaving
        ctx = original(self, body, dialect)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "turns": turns, "est_in": ctx.est_tokens_in, "est_out": ctx.est_tokens_out,
                "growth": round(growth), "l0": round(l0), "threshold": int(threshold), "compacted": bool(ctx.compacted),
            }) + "\n")
        return ctx

    cliff_engine.Engine.prepare = prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="strong", help="price-table entry the agent runs on")
    ap.add_argument("--extra-steps", type=float, default=2.0)
    ap.add_argument("--ceiling", type=int, default=52000)
    ap.add_argument("--min-gap", type=int, default=4)
    ap.add_argument("--log", default=str(ROOT / "runs" / "cliff_eoq_thresholds.jsonl"))
    ap.add_argument("cliff_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    install(args.model, args.extra_steps, Path(args.log), args.ceiling, args.min_gap)
    rest = args.cliff_args[1:] if args.cliff_args[:1] == ["--"] else args.cliff_args
    sys.exit(cliff_cli.main(rest))


if __name__ == "__main__":
    main()
