"""Circuit breaker for the cloud model.

Without it, every question during an outage first waits for the SDK's retries
(connection errors, timeouts) before it falls back. After THRESHOLD failures in a
row the breaker "opens": questions go straight to the local model for COOLDOWN_S
seconds. Then ONE question may try Claude again ("half-open"): success closes the
breaker, failure opens it for another cooldown.

The state is a small JSON file, so separate CLI runs share it. If the file is
missing or broken the breaker is closed: it can never block Claude by mistake.
"""

import json
import time
from pathlib import Path

THRESHOLD = 3     # failures in a row before the breaker opens
COOLDOWN_S = 60   # seconds to skip Claude after it opens
STATE_FILE = Path("data/processed/circuit.json")


class CircuitBreaker:
    def __init__(self, path: Path | None = None, threshold: int = THRESHOLD,
                 cooldown_s: float = COOLDOWN_S, clock=time.time):
        self.path = path  # None: keep the state in memory only (tests, one process)
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock  # tests pass a fake clock
        self._state = {"failures": 0, "opened_at": None}

    # --- state

    def _load(self) -> dict:
        if self.path is None:
            return dict(self._state)
        try:
            s = json.loads(self.path.read_text())
            return {"failures": int(s["failures"]), "opened_at": s["opened_at"]}
        except (OSError, ValueError, KeyError, TypeError):
            return {"failures": 0, "opened_at": None}  # unknown state: closed

    def _save(self, s: dict) -> None:
        self._state = s
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(s))
            tmp.replace(self.path)  # atomic: a reader never sees half a file

    def state(self) -> str:
        s = self._load()
        if s["opened_at"] is None:
            return "closed"
        if self.clock() - s["opened_at"] < self.cooldown_s:
            return "open"
        return "half_open"

    # --- used by the agent

    def allow(self) -> bool:
        """May this question try Claude?"""
        return self.state() != "open"

    def record_success(self) -> None:
        if self._load() != {"failures": 0, "opened_at": None}:
            self._save({"failures": 0, "opened_at": None})

    def record_failure(self) -> None:
        s = self._load()
        s["failures"] += 1
        if s["failures"] >= self.threshold:
            s["opened_at"] = self.clock()  # open, or open again after a half-open try
        self._save(s)

    def seconds_left(self) -> int:
        s = self._load()
        if s["opened_at"] is None:
            return 0
        return max(0, round(self.cooldown_s - (self.clock() - s["opened_at"])))
