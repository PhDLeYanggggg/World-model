"""Replay the frozen dual-clock run with JSON-safe diagnostic counters."""

from collections import Counter
from pathlib import Path
import platform
import sys
from unittest.mock import patch

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_dual_clock as frozen


def main():
    if len(sys.argv) != 1:
        raise SystemExit("This recovery entry point always replays without training")
    original = frozen.evaluate

    def encode(*args, **kwargs):
        z = original(*args, **kwargs)
        # NumPy counts are produced by comparing query indices; preserve their numeric values.
        z["diagnostics"] = Counter({k: int(v) for k, v in z["diagnostics"].items()})
        return z

    run = frozen.run
    parent = frozen.OUT/"registration.json"
    heads = frozen.OUT/"head_freeze.json"
    if run.read(heads)["registration"] != run.sha(parent):
        raise ValueError("Require completed checkpoint freeze from the original run")
    r = frozen.OUT/"serialization_recovery_v2.json"
    abandoned = frozen.OUT/"serialization_recovery.json"
    run.once(r, run.stable(dict(code=run.artifact(Path(__file__)), registration=run.artifact(parent),
        heads=run.artifact(heads), abandoned_dict_conversion=run.artifact(abandoned),
        reason="normalize NumPy int64 diagnostics before JSON serialization without changing Counter aggregation",
        no_policy_or_threshold_change=True, replay_only=True, original_2pct=True, confirmation_access=False)))
    with patch.object(frozen, "evaluate", encode), patch.object(sys, "argv", [sys.argv[0], "--replay"]):
        frozen.main()


if __name__ == "__main__":
    main()
