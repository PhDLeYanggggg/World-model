"""Export anonymous pedestrian budget, capacity, and paired-debit summaries."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_sparse_support_replay as base

ROOT, sha = base.ROOT, base.sha
LEDGER = base.SOURCE.parent
CAPACITY = LEDGER/"capacity_factorial_v1"
POLICY = LEDGER/"budget_risk_v1"
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/moving_budget_replay_v1/anonymous_moving_budget.zip"


def inputs():
    def read(path):
        return json.loads(path.read_text())

    docs = []
    for root, name, reg in ((CAPACITY, "analysis", "registration"), (POLICY, "analysis", "registration"),
                            (POLICY, "attribution", "attribution_registration")):
        f = root/(reg+".json")
        a = read(root/(name+".json"))
        if a["registration"] != sha(f) or a["confirmation_access"] or not a["original_2pct"]:
            raise ValueError("Unbound or disallowed budget evidence")
        for p in read(f)["parents"]:
            if sha(ROOT/p["path"]) != p["sha256"]:
                raise ValueError("Changed budget experiment parent")
        docs.append(a)
    if docs[1]["heads"] != sha(POLICY/"head_freeze.json"):
        raise ValueError("Changed budget head receipt")
    return docs


def build(out):
    a, b, c = inputs()
    cells = [{(s, r["seed"]): r for s, rows in z["by_site"].items() for r in rows} for z in (a, b, c)]
    if set(cells[0]) != set(cells[1]) or set(cells[0]) != set(cells[2]):
        raise ValueError("Require identical source/seed rosters")
    groups = []
    for s in sorted(b["by_site"]):
        g = []
        for r in sorted(b["by_site"][s], key=lambda z: z["seed"]):
            key = s, r["seed"]
            debit = {f"{k}_{f}": v for k, z in cells[2][key]["effects"].items() for f, v in z.items()}
            g.append(dict(metrics=r["metrics"], effects={**r["effects"], **debit},
                          counts=cells[0][key]["counts"], capacity={}, bridge={}))
        groups.append(g)
    effects = dict(b["summary"]["effects"])
    effects.update({f"{k}_{f}": v for k, z in c["summary"].items() for f, v in z.items()})
    expected = dict(metrics=b["summary"]["metrics"], effects=effects,
                    counts=a["summary"]["counts"], capacity={}, bridge={})
    base.package(out, dict(groups=groups), expected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
