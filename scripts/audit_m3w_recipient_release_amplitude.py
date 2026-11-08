"""Separate full-strength straight paths from curvature at free count."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_recipient_release_capacity as base


run, np = base.run, base.np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", action="store_true")
    a = cli.parse_args()
    binding, identity, cut = base.register()
    path = base.OUT/"analysis.json"
    parent = run.read(path)
    if parent["registration"] != binding or parent["calibration_access"] or parent["independent_confirmation"]:
        raise ValueError("Require fitting-only capacity analysis")
    receipt = run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(path), width=8))
    rows = []
    for rec, g, cv, ols, bank, y, valid, inv, _, w in base.fitting_blocks(identity, lambda **_: None):
        cache = base.PRIVATE/f"full_free_{rec}.json"
        bound = run.read(base.PRIVATE/f"capacity_{rec}.json")["menus"]
        if not cache.exists():
            if a.replay:
                raise ValueError("Replay cannot optimize")
            val = base.capacity.assess(cv, ols, bank[:, :8], y, valid, inv, None, w, rec, cut)
            if (bound["curved_free"]["status"] in ("risk_infeasible", "negative_upper")
                    and val["status"] == "verified_both"):
                raise ValueError("Sub-bank positive contradicts expanded-bank bound")
            run.once(cache, run.stable(dict(receipt=receipt, result=val)))
        old = run.read(cache)
        if old["receipt"] != receipt:
            raise ValueError("Unbound full-strength control")
        val = old["result"]
        if val["status"] == "verified_both":
            rr, aa = (np.array(x, int) for x in val["witness"])
            actual = base.capacity.evidence(cv, ols, bank[:, :8], y, valid, inv, None, w, rec, cut, rr, aa)
            if actual != val["evidence"] or not actual["risk"]["supported"]:
                raise ValueError("Full-strength witness did not replay")
        rows.append(dict(recording=rec, source=g, full_free={k: v for k, v in val.items() if k != "witness"},
                         curved_free=bound["curved_free"]["status"]))
    counts = Counter(r["full_free"]["status"] for r in rows)
    run.once(base.OUT/"amplitude.json", run.stable(dict(receipt, rows=rows,
        summary={k: counts[k] for k in base.capacity.STATES}, calibration_access=False,
        confirmation_access=False, deployment=False, outcome_aware_only=True)))


if __name__ == "__main__":
    main()
