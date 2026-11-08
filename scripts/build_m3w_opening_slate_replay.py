"""Reproduce opening-slate counts and paired intervals from anonymous scalars."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_radius_opening_replay as parent

base = parent.base
SOURCE = parent.SOURCE.parent/"opening_slate_v2"
DEFAULT = base.ROOT/"data/stage_cvpr2027_experiments/moving_opening_slate_replay_v2/anonymous_slate.zip"


def inputs():
    parent.inputs()
    reg, analysis, heads = (SOURCE/f"{k}.json" for k in ("registration", "analysis", "head_freeze"))
    a, r, h = (json.loads(p.read_text()) for p in (analysis, reg, heads))
    if (a["registration"] != base.sha(reg) or a["heads"] != base.sha(heads)
            or h["registration"] != a["registration"] or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require frozen opening-slate results")
    for p in [r["code"], r["model"], *r["parents"]]:
        if base.sha(base.ROOT/p["path"]) != p["sha256"]:
            raise ValueError("Changed opening-slate lineage")
    return a


def build(out):
    a = inputs()
    groups = [[dict(metrics=r["metrics"], counts=r["recall"], effects=r["effects"], capacity={}, bridge={})
               for r in sorted(rows, key=lambda z: z["seed"])] for _, rows in sorted(a["by_site"].items())]
    base.package(out, dict(groups=groups), dict(metrics=a["metrics"], effects=a["effects"],
        counts={k: v for k, v in a["recall"].items() if v}, capacity={}, bridge={}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
