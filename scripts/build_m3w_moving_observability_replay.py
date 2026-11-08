"""Export anonymous paired observability comparisons from the 14 sources."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_sparse_support_replay as base

SOURCE = base.SOURCE.parent/"observability_v1"
DEFAULT = base.ROOT/"data/stage_cvpr2027_experiments/moving_observability_replay_v1/anonymous_observability.zip"


def inputs():
    a = json.loads((SOURCE/"analysis.json").read_text())
    p = SOURCE/"registration.json"
    if (a["registration"] != base.sha(p) or a["heads"] != base.sha(SOURCE/"head_freeze.json")
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require frozen observability results")
    for x in json.loads(p.read_text())["parents"]:
        if base.sha(base.ROOT/x["path"]) != x["sha256"]:
            raise ValueError("Changed observability parent")
    return a


def build(out):
    a = inputs()
    groups = [[dict(metrics=r["metrics"], effects={**r["effects"], **{"brier_"+k: v for k, v in r["brier"].items()}},
                    counts={}, capacity={}, bridge={})
               for r in sorted(rows, key=lambda z: z["seed"])]
              for _, rows in sorted(a["by_site"].items())]
    s = a["summary"]
    effects = {**s["effects"], **{"brier_"+k: v for k, v in s["brier"].items()}}
    base.package(out, dict(groups=groups),
                 dict(metrics=s["metrics"], effects=effects, counts={}, capacity={}, bridge={}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
