"""Verify matched-selector counts and descriptive site bootstrap intervals."""

import hashlib
import json
from pathlib import Path

import numpy as np


def aggregate(studies):
    result = {}
    draw = np.random.default_rng(73107).integers(3, size=(3000, 3))
    for name, groups in studies.items():
        if len(groups) != 3 or any(len(g) != 9 for g in groups):
            raise ValueError("Need three intersections with nine repeated views each")
        flat = [r for g in groups for r in g]
        metrics = {k: [sum(r["counts"][k][i] for r in flat) for i in range(3)] for k in flat[0]["counts"]}
        effects = {}
        for k in flat[0]["effects"]:
            v = np.array([np.mean([r["effects"][k] for r in g]) for g in groups])
            effects[k] = [float(v.mean()), *np.quantile(v[draw].mean(1), [.025, .975]).tolist()]
        result[name] = dict(metrics=metrics, effects=effects)
    return result


def verify(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Changed study roster")
    for name, a in actual.items():
        b = expected[name]
        if a["metrics"] != b["metrics"] or set(a["effects"]) != set(b["effects"]):
            raise ValueError("Changed count or contrast roster")
        for k, v in a["effects"].items():
            np.testing.assert_allclose(v, b["effects"][k], atol=1e-12, rtol=0.)


def main():
    root = Path(__file__).resolve().parent
    read = lambda name: json.loads((root/name).read_text())
    manifest = read("manifest.json")
    if set(manifest) != {"README.md", "requirements.txt", "replay.py", "scalars.json", "expected.json"}:
        raise ValueError("Unexpected archive layout")
    for k, v in manifest.items():
        if hashlib.sha256((root/k).read_bytes()).hexdigest() != v:
            raise ValueError("Changed archive artifact")
    result = aggregate(read("scalars.json"))
    verify(result, read("expected.json"))
    print(json.dumps(dict(verified=True, study_radius_pairs=len(result), intersections=3, repeated_views=27,
                          intervals=sum(len(v["effects"]) for v in result.values()))))


if __name__ == "__main__":
    main()
