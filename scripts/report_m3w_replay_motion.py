"""Serialize and replay the frozen motion experiment's numerical readout."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_replay_motion as fit

REG = fit.PUBLIC / "readout_registration.json"
FILES = (Path(__file__), fit.PUBLIC / "readout_amendment.md",
         ROOT / "tests/test_m3w_replay_motion_readout.py")


def serializable(value):
    if isinstance(value, (fit.np.ndarray, fit.np.generic)):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "score", "replay"))
    args = p.parse_args()
    cut, jobs = fit.register()
    run = fit.run
    run.once(REG, dict(bindings=[run.artifact(f) for f in (*FILES, fit.REG,
        fit.PUBLIC / "train_execution.json")], change="JSON representation only"))
    if args.phase == "register":
        return
    for path in (*FILES, REG, *fit.FILES, fit.REG):
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError("Commit the readout-only amendment before replay")
    fit.torch.set_num_threads(fit.CFG["torch_threads"])
    fit.torch.set_num_interop_threads(1)
    with (fit.PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        receipt = fit.PUBLIC / f"{args.phase}_execution.json"
        if receipt.exists():
            fit.prev.receipt(fit.PUBLIC, args.phase)
            return
        if args.phase == "replay":
            fit.prev.receipt(fit.PUBLIC, "score")
        frozen = fit.prev.receipt(fit.PUBLIC, "train")
        out = fit.PUBLIC / "analysis.json"
        result = fit.score(jobs, cut, frozen, lambda **values: print(
            dict(pid=os.getpid(), phase=args.phase, **values), flush=True))
        run.once(out, run.stable(serializable(result)))
        run.once(receipt, dict(pid=os.getpid(), seconds=time.monotonic() - start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
