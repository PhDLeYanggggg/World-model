"""Standalone numerical replay of aggregate deferral evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")


def validate(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for name, digest in manifest.items():
        file = root/name
        if not file.resolve().is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected replay member: {name}")


def replay(cells):
    if len(cells) != 7 or any(sorted(r["seed"] for r in v) != [1, 2, 3] for v in cells):
        raise ValueError("Require three paired seed views per development location")
    flat = [r for v in cells for r in v]
    draw = np.random.default_rng(817).integers(len(cells), size=(3000, len(cells)))
    direct = {}
    for k in flat[0]["direct"]:
        a = np.array([np.mean([r["direct"][k] for r in v]) for v in cells])
        direct[k] = [float(a.mean()), *np.quantile(a[draw].mean(1), [.025, .975]).tolist()]
    arms = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in FIELDS} for k in flat[0]["metrics"]}
    oracle = {k: sum((Counter(r["oracle"][k]) for r in flat), Counter()) for k in flat[0].get("oracle", {})}
    if any(a["views"] != 606 for a in arms.values()) or any(sum(v.values()) != 606 for v in oracle.values()):
        raise ValueError("A required recording-seed view disappeared")
    return dict(arms=arms, direct=direct, oracle=oracle)


def match(actual, expected):
    if actual["arms"] != expected["arms"] or actual["oracle"] != expected["oracle"]:
        raise ValueError("Recording totals or solver statuses changed")
    if set(actual["direct"]) != set(expected["direct"]):
        raise ValueError("Comparison roster changed")
    for key, values in expected["direct"].items():
        np.testing.assert_allclose(actual["direct"][key], values, atol=1e-13, rtol=0.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    validate(root)
    src = json.loads((root/"statistics.json").read_text())
    expected = json.loads((root/"expected.json").read_text())
    if set(src) != set(expected):
        raise ValueError("Study roster changed")
    out = {}
    for name, cells in src.items():
        out[name] = replay(cells)
        match(out[name], expected[name])
    if a.output:
        a.output.write_text(json.dumps(out, sort_keys=True, indent=2)+"\n")
    print(json.dumps(dict(studies=len(out), contrasts=sum(len(v["direct"]) for v in out.values()),
        result="recomputed_paired_bootstrap_and_view_totals", independent_confirmation=False), sort_keys=True))


if __name__ == "__main__":
    main()
