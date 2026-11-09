"""Count how often the frozen selectors actually choose the added EqMotion shape."""

from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_eqmotion as run

np, av, study, base = run.np, run.av, run.study, run.base
ARMS = ("eq_additive", "eq_joint")


def new_shapes(path, d, eq):
    """A changed agent counts only if its path could not come from the old menu."""
    c, b = d["cv"], run.expanded(d, eq)
    delta = b-c[:, None]
    native = av.motion.peak(delta)
    radius = av.motion.peak(path-c)
    legal = (native > 0.) & (native >= radius[:, None]-1e-10)
    factor = np.divide(radius[:, None], native, out=np.zeros_like(native), where=legal).clip(max=1.)
    exact = np.isclose(path[:, None], c[:, None]+factor[..., None, None]*delta, atol=1e-9, rtol=1e-9).all((2, 3))
    membership = exact & legal
    edited = radius > 0.
    if not membership[edited].any(1).all():
        raise ValueError("Changed agent path is outside the exact peak menu")
    return edited & membership[:, -1] & ~membership[:, :-1].any(1)


def main():
    key, previous, source, _, _, _ = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    f = study.read(study.checked(result["freeze"]))
    if result["registration"] != key or f["registration"] != key or f["held_labels_used"]:
        raise ValueError("Require the scored frozen forecaster")
    data = run.scale.read_inputs(previous, source)
    by_city = defaultdict(Counter)
    unique = {a: set() for a in ARMS}
    for fold in f["folds"].values():
        city, seed = (fold["identity"][k] for k in ("held", "seed"))
        with np.load(study.checked(fold["forecast"]), allow_pickle=False) as z:
            bank = dict(z)
        for n, lo, hi in zip(bank["names"], bank["offsets"][:-1], bank["offsets"][1:], strict=True):
            d = data[n]
            for a, old in zip(ARMS, ("additive", "relative_joint"), strict=True):
                v = bank[a][lo:hi]
                new = int(new_shapes(v, d, bank["raw_eq"][lo:hi]).sum())
                if new:
                    unique[a].add(str(n))
                by_city[city].update({f"{a}_forecast_changed": int(not np.array_equal(v, d[f"{old}_{seed}"])),
                    f"{a}_novel_shape_edits": new, f"{a}_novel_shape_views": int(new > 0)})
    total = dict(sum(by_city.values(), Counter()))
    out = dict(parent=study.artifact(p), code=study.artifact(Path(__file__)), totals=total,
        unique_scenarios={a: len(v) for a, v in unique.items()}, by_city={k: dict(v) for k, v in by_city.items()},
        result_source="fresh_run", outcome_labels_used=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"usage.json", out)
    print(study.json.dumps({k: out[k] for k in ("totals", "unique_scenarios")}, indent=2))


if __name__ == "__main__":
    main()
