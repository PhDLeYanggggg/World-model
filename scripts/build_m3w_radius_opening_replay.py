"""Export anonymous source-level scalars for the matched opening study."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_small_support_exchange_replay as parent

base = parent.base
SOURCE = parent.SOURCE.parent/"radius_opening_v1"
DEFAULT = base.ROOT/"data/stage_cvpr2027_experiments/moving_radius_opening_replay_v1/anonymous_opening.zip"


def inputs():
    parent.inputs()
    reg = SOURCE/"registration.json"
    a, r = (json.loads(p.read_text()) for p in (SOURCE/"analysis.json", reg))
    if a["registration"] != base.sha(reg) or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require frozen radius-opening results")
    for p in [r["code"], r["model"], *r["parents"]]:
        if base.sha(base.ROOT/p["path"]) != p["sha256"]:
            raise ValueError("Changed opening evidence")
    return a


def build(out):
    a = inputs()
    groups = [[dict(metrics=r["metrics"], counts=r["oracle"], effects=r["effects"], capacity={}, bridge={})
               for r in sorted(rows, key=lambda z: z["seed"])] for _, rows in sorted(a["by_site"].items())]
    base.package(out, dict(groups=groups), dict(metrics=a["metrics"], effects=a["effects"],
        counts={k: v for k, v in a["oracle"].items() if v}, capacity={}, bridge={}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
