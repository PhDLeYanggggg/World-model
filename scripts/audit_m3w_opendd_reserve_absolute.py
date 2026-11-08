"""Reconstruct direct CV gains from the disjoint incumbent and reserve edits."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_candidate_replay as source

OUT = source.REPORTS/"opendd_reserve_v1"


def summarize(a):
    cells = [z for _, z in sorted(a["by_site"].items())]
    if len(cells) != 7 or any(len(z) != 3 for z in cells):
        raise ValueError("Expected all sites and seeds")
    draw = np.random.default_rng(817).integers(len(cells), size=(3000, len(cells)))
    result = {}
    for arm in ("base", "easy", "gain", "joint", "easy_same_query", "gain_same_query"):
        s = []
        for group in cells:
            values = []
            for c in group:
                e = c["effects"]
                value = e["base_vs_cv"]+e.get(f"{arm}_vs_base", 0.)
                if arm == "joint":
                    np.testing.assert_allclose(value, e["joint_vs_cv"], atol=1e-12, rtol=0.)
                values.append(value)
            s.append(np.mean(values))
        x = np.asarray(s)
        result[arm] = dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                           worst_site=float(x.min()))
    return result


def main():
    a = source.read(OUT.name)
    z = source.encoded(dict(analysis=source.sha(OUT/"analysis.json"), original_2pct=True, confirmation_access=False,
        disjoint_edits=True, effects_vs_cv=summarize(a)))
    f = OUT/"absolute.json"
    if f.exists() and f.read_bytes() != z:
        raise ValueError("Frozen absolute readout changed")
    if not f.exists():
        with f.open("xb") as g:
            g.write(z)
    print(f, flush=True)


if __name__ == "__main__":
    main()
