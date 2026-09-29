"""Spend limits enforced by the proxy: per trajectory and across all runs.

When a limit is reached the proxy answers the agent's next call itself with
a "task complete" message instead of calling a model, so the trial ends
normally and the verifier scores whatever state the agent left. The same
rule applies to every policy, so it caps runaway loops without favouring one.
"""

from __future__ import annotations

import glob
import json
import threading
from collections import defaultdict
from pathlib import Path

# Terminus-2's action format; it asks for confirmation after task_complete, which gets the same answer.
TERMINUS_STOP = json.dumps(
    {
        "analysis": "Budget for this trajectory is exhausted; stopping here.",
        "plan": "",
        "commands": [],
        "task_complete": True,
    }
)


def spent_in_logs(pattern: str) -> float:
    total = 0.0
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if not str(rec.get("model", "")).startswith("mock"):  # offline test pool, never billed
                        total += rec.get("cost") or 0.0
    return total


class Budget:
    def __init__(
        self,
        *,
        per_session_usd: float | None = None,
        total_usd: float | None = None,
        spent_before: float = 0.0,
        stop_response: str = TERMINUS_STOP,
    ) -> None:
        self.per_session_usd = per_session_usd
        self.total_usd = total_usd
        self.stop_response = stop_response
        self._total = spent_before
        self._by_session: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict, *, config_dir: str) -> Budget:
        ledger = cfg.get("ledger_glob", "../runs/*/calls.jsonl")
        if not Path(ledger).is_absolute():
            ledger = str(Path(config_dir) / ledger)
        return cls(
            per_session_usd=cfg.get("per_session_usd"),
            total_usd=cfg.get("total_usd"),
            spent_before=spent_in_logs(ledger) if cfg.get("total_usd") is not None else 0.0,
            stop_response=cfg.get("stop_response", TERMINUS_STOP),
        )

    def exhausted(self, session_id: str) -> str | None:
        """Why this session must stop, or None."""
        with self._lock:
            if self.total_usd is not None and self._total >= self.total_usd:
                return f"total budget ${self.total_usd:.2f} spent"
            if self.per_session_usd is not None and self._by_session[session_id] >= self.per_session_usd:
                return f"trajectory budget ${self.per_session_usd:.2f} spent"
            return None

    def charge(self, session_id: str, usd: float) -> None:
        with self._lock:
            self._by_session[session_id] += usd
            self._total += usd

    @property
    def total_spent(self) -> float:
        return self._total
