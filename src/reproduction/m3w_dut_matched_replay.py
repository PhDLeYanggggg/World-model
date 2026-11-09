"""Recompute the DUT evidence table from anonymous two-site scalar values."""

import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(sites):
    if len(sites) != 2 or any(len(group) != 3 for group in sites):
        raise ValueError("Expected two physical sites and three paired seeds")
    flat = [v for group in sites for v in group]
    d = np.random.default_rng(917).integers(2, size=(3000, 2))
    pairs, arms = {}, {}
    for k in flat[0]["paired"]:
        pairs[k] = {}
        for f in flat[0]["paired"][k]:
            x = np.array([np.mean([v["paired"][k][f] for v in group]) for group in sites])
            pairs[k][f] = dict(point=float(x.mean()), sites=x.tolist(),
                               interval95=np.quantile(x[d].mean(1), [.025, .975]).tolist())
    for k in flat[0]["arms"]:
        z = [v["arms"][k] for v in flat]
        arms[k] = dict(complete_ade=float(np.mean([v["ade"] for v in z])),
                      endpoint_fde=float(np.mean([v["fde"] for v in z])),
                      lower_pp=float(np.mean([v["lower_pp"] for v in z])),
                      worst_easy_pct=max((v["easy"] for v in z if v["easy"] is not None), default=None),
                      edits=sum(v["edits"] for v in z),
                      undefined_selected_easy=sum(v["undefined_selected_easy"] for v in z))
    return dict(arms=arms, paired=pairs, strict=sum(v["strict"] for v in flat), common=sum(v["common"] for v in flat))


def verify(a, b):
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        for k in a:
            verify(a[k], b[k])
    elif isinstance(a, list) and isinstance(b, list):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.)
    elif isinstance(a, float) and isinstance(b, float):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.)
    elif a != b:
        raise ValueError("Scalar readout changed")


def main():
    p = Path(__file__).resolve().parent
    manifest = json.loads((p / "manifest.json").read_text())
    expected = {"README.md", "requirements.txt", "replay.py", "scalars.json", "expected.json"}
    if set(manifest) != expected:
        raise ValueError("Unexpected anonymous archive contents")
    for k, digest in manifest.items():
        if hashlib.sha256((p / k).read_bytes()).hexdigest() != digest:
            raise ValueError("Changed artifact: " + k)
    a = summarize(json.loads((p / "scalars.json").read_text()))
    verify(a, json.loads((p / "expected.json").read_text()))
    print(json.dumps(dict(verified=True, arm_tables=len(a["arms"]),
                          pair_intervals=sum(len(v) for v in a["paired"].values()), sites=2)))


if __name__ == "__main__":
    main()
