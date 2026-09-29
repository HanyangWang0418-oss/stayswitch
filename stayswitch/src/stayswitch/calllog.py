"""Append-only JSONL log of every routed call; the unit of all later analysis."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class CallLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


def read_calls(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
