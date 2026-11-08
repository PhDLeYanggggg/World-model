"""Recompute pedestrian sparse-support tables from anonymous source scalars."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def aggregate(data):
    groups = data["groups"]
    if len(groups) != 14 or any(len(g) != 3 for g in groups):
        raise ValueError("Require 14 sources with three repeated seed views")
    rows = [z for g in groups for z in g]
    counts = {k: sum((Counter(r[k]) for r in rows), Counter())
              for k in ("counts", "capacity", "bridge")}
    metrics = {k: {f: sum(r["metrics"][k][f] for r in rows) for f in m}
               for k, m in rows[0]["metrics"].items()}
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    effects = {}
    for k in rows[0]["effects"]:
        x = np.asarray([np.mean([r["effects"][k] for r in g]) for g in groups])
        effects[k] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())
    return dict(metrics=metrics, effects=effects, **counts)


def verify(actual, expected):
    for k in ("metrics", "counts", "capacity", "bridge"):
        if actual[k] != expected[k]:
            raise ValueError("Sparse support count table changed: "+k)
    if actual["effects"].keys() != expected["effects"].keys():
        raise ValueError("Sparse support contrast roster changed")
    for k, a in actual["effects"].items():
        b = expected["effects"][k]
        np.testing.assert_allclose(a["point"], b["point"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(a["interval95"], b["interval95"], atol=1e-12, rtol=0.)


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/"manifest.json").read_text())
    if set(manifest) != {"README.md", "expected.json", "statistics.json", "replay.py", "requirements.txt"}:
        raise ValueError("Unexpected package manifest")
    for k, h in manifest.items():
        if hashlib.sha256((root/k).read_bytes()).hexdigest() != h:
            raise ValueError("Replay artifact changed: "+k)
    data = json.loads((root/"statistics.json").read_text())
    expected = json.loads((root/"expected.json").read_text())
    a = aggregate(data)
    verify(a, expected)
    print(json.dumps(dict(verified=True, source_groups=14, repeated_views=102,
                          policy_rows=len(a["metrics"]), intervals=len(a["effects"]))))


if __name__ == "__main__":
    main()
