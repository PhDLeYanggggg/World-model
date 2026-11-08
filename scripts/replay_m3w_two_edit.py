"""Run or recover the registered study with the short-roster aggregation fix."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_two_edit as prior

run = prior.run


def normalize(result):
    short = result["short"]
    if all(v is None for v in short.values()):
        result["short"] = None
    elif any(v is None for v in short.values()):
        raise ValueError("Two-edit arms must share the short roster")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="Allow initial or resumed fitting; default is score-only.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    prior.OUT.mkdir(parents=True, exist_ok=True)
    prior.PRIVATE.mkdir(parents=True, exist_ok=True)
    prior.fit.PRIVATE = prior.PRIVATE
    start = time.monotonic()
    with (prior.PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = prior.register()
        reg = prior.OUT / "readout_repair_v2.json"
        run.once(reg, run.stable(dict(repair=run.artifact(Path(__file__)), registration=inputs[0],
            change="use null for the entire absent short-roster cell rather than a dictionary of null arms",
            predictions_changed=False, training_only_with_explicit_fit_flag=True)))
        evaluate = prior.evaluate

        def beat(**values):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **values)
            run.kin.prior.source.atomic(prior.PRIVATE / "heartbeat.json", z)
            print(z, flush=True)

        try:
            prior.evaluate = lambda *args: normalize(evaluate(*args))
            updates = prior.execute(*inputs, not args.fit, beat)
            if not args.fit and updates != 0:
                raise ValueError("Recovery must not train")
        finally:
            prior.evaluate = evaluate
        beat(phase="complete", updates=updates)


if __name__ == "__main__":
    main()
