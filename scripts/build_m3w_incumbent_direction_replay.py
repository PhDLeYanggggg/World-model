"""Reconstruct direction and attribution statistics from anonymous source scalars."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_moving_observability_replay as old

base = old.base
SOURCE = old.SOURCE.parent / "incumbent_direction_v2"
ATTR = old.SOURCE / "cost_attribution_v1"
DEFAULT = base.ROOT / "data/stage_cvpr2027_experiments/moving_incumbent_direction_replay_v2/anonymous_direction.zip"


def inputs():
    old.inputs()
    docs = []
    for folder in (SOURCE, ATTR):
        reg = folder / "registration.json"
        a = json.loads((folder / "analysis.json").read_text())
        if a["registration"] != base.sha(reg) or a["confirmation_access"] or not a["original_2pct"]:
            raise ValueError("Require original registered evaluation")
        for p in json.loads(reg.read_text())["parents"]:
            if base.sha(base.ROOT / p["path"]) != p["sha256"]:
                raise ValueError("Changed parent evidence")
        docs.append(a)
    if docs[0]["heads"] != base.sha(SOURCE / "head_freeze.json"):
        raise ValueError("Changed direction freeze")
    return docs


def flatten(values, prefix=""):
    return {f"{prefix}{k}_{f}": v for k, row in values.items() for f, v in row.items()}


def build(out):
    a, b = inputs()
    if set(a["by_site"]) != set(b["by_site"]):
        raise ValueError("Mismatched source populations")
    groups = []
    for s in sorted(a["by_site"]):
        x, y = ({r["seed"]: r for r in z["by_site"][s]} for z in (a, b))
        if set(x) != set(y):
            raise ValueError("Mismatched seed populations")
        groups.append([dict(metrics=x[i]["metrics"],
                            effects={**flatten(x[i]["effects"]), **flatten(y[i]["effects"], "attr_")},
                            counts=y[i]["counts"], capacity={}, bridge={}) for i in sorted(x)])
    base.package(out, dict(groups=groups),
                 dict(metrics=a["summary"]["metrics"], counts=b["counts"], capacity={}, bridge={},
                      effects={**flatten(a["summary"]["effects"]), **flatten(b["summary"], "attr_")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
