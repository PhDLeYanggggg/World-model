"""Package the OpenDD completion-loss contrasts without trajectories or paths."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_candidate_replay as base

NAME = "opendd_completion_forecast_v1"
DEFAULT = base.ROOT / "data/stage_cvpr2027_experiments" / NAME / "anonymous_completion.zip"


def collect():
    a = base.read(NAME)
    report = base.REPORTS / NAME
    f = report / "freeze.json"
    b = json.loads((report / "radius_analysis.json").read_text())
    if (a["freeze"] != base.sha(f) or
            b["parent"]["sha256"] != base.sha(report / "analysis.json") or
            not b["original_2pct"] or b["confirmation_access"]):
        raise ValueError("Unbound completion-loss readout")
    s = a["summary"]
    effects = {}
    for k, v in s["raw"].items():
        for r, e in v["lower"].items():
            effects[f"raw_{k}_{r}"] = e
    for k, v in s["direct_vs_masked"].items():
        effects.update({f"direct_{k}_{m}": e for m, e in v.items()})
    effects.update({f"matched_{k}": v["direct_vs_anchor"] for k, v in s["matched"].items()})
    counts = {f"raw_{k}": v["record"] for k, v in s["raw"].items()}
    counts.update({f"matched_{k}": v["record"] for k, v in s["matched"].items()})

    def row(v):
        full, matched = v["full"], v["matched"]
        e = {f"raw_{k}_{r}": e for k, x in full["arms"].items() for r, e in x["lower"].items()}
        e.update({f"direct_{k}_{m}": e for k, x in full["effects"].items() for m, e in x.items()})
        e.update({f"matched_{k}": x for k, x in matched["gains_vs_anchor"].items()})
        c = {f"raw_{k}": x["record"] for k, x in full["arms"].items()}
        c.update({f"matched_{k}": x for k, x in matched["policies"].items()})
        return base.row(e, c)

    main = base.pack(a, lambda v: v, row, effects, counts)
    s = b["summary"]
    effects = {f"{k}_{p}": e for k, v in s.items() for p, e in v.items() if p not in ("controls", "slices")}
    counts = {f"{k}_{p}": c for k, v in s.items() for p, c in v["controls"].items()}

    def radius_row(v):
        e = {f"{k}_{p}": e for k, x in v["arms"].items() for p, e in x["direct"].items()}
        c = {f"{k}_{p}": c for k, x in v["arms"].items() for p, c in x["controls"].items()}
        return base.row(e, c)

    radius = base.pack({"by_site": b["rows"]}, lambda v: v, radius_row, effects, counts)
    names = ("completion_objectives", "realized_peak_control")
    return dict(studies=dict(zip(names, (main[0], radius[0]), strict=True)), risk_cap=.02,
                independent_confirmation=False), dict(zip(names, (main[1], radius[1]), strict=True))


def build(out):
    packet, expected = collect()
    base.replay.check(base.replay.recompute(packet), expected)
    files = {
        "README.md": b"Run python replay.py with pinned NumPy. Anonymous OpenDD development scalars only.\n",
        "replay.py": Path(base.replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={base.np.__version__}\n".encode(),
        "statistics.json": base.encoded(packet),
        "expected.json": base.encoded(expected),
    }
    files["manifest.json"] = base.encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = out.with_suffix(".pending")
    with ZipFile(pending, "w") as z:
        for k, value in sorted(files.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), value,
                       compress_type=ZIP_DEFLATED, compresslevel=9)
    pending.replace(out)
    print(base.sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
