"""Anonymous scalar replay of the TUMDOT peer and replacement-budget comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_tumdot_selector_replay as base

ROOT, REPORTS, replay = base.ROOT, base.REPORTS, base.replay
NAMES = ("tumdot_peer_choice_v1", "tumdot_peer_budget_v1")
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/tumdot_peer_replay_v1/anonymous_peer.zip"


def inputs():
    base.inputs()
    result = []
    for name in NAMES:
        out = REPORTS/name
        reg = out/"registration.json"
        a, r = (json.loads(p.read_text()) for p in (out/"analysis.json", reg))
        if (a["registration"] != base.sha(reg) or not a["original_2pct"] or a["confirmation_access"]
                or not a["source_is_previously_exposed"]):
            raise ValueError("Require frozen exposed peer comparison")
        if "heads" in a and a["heads"] != base.sha(out/"head_freeze.json"):
            raise ValueError("Changed peer checkpoint receipt")
        for z in [r["code"], *([r["model"]] if "model" in r else []), *r["parents"]]:
            if base.sha(ROOT/z["path"]) != z["sha256"]:
                raise ValueError("Changed peer experiment lineage")
        result.append(a)
    return result


def build(out):
    studies, expected = {}, {}
    for label, a in zip(("peer", "matched"), inputs(), strict=True):
        rows = a["rows"]
        sites = sorted({k.split("-")[0] for k in rows})
        for part, v in a["summary"].items():
            key = f"{label}_{part}"
            studies[key] = [[base.row(z["scores"][part]) for k, group in sorted(rows.items())
                             if k.startswith(s+"-") for z in group] for s in sites]
            counts = {k: [m[f] for f in ("risk_pass", "joint_cv", "joint_cv_ols")] for k, m in v["metrics"].items()}
            effects = {f"{k}_{n}": f[n] for k, f in v["effects"].items() for n in ("lower", "observed")}
            effects.update({f"{k}_{n}": m[n] for k, m in v["metrics"].items() for n in ("cv", "ols4")})
            expected[key] = dict(metrics=counts,
                                 effects={k: [f["point"], *f["interval95"]] for k, f in effects.items()})
    replay.verify(replay.aggregate(studies), expected)
    files = {
        "scalars.json": base.encoded(studies), "expected.json": base.encoded(expected),
        "README.md": b"Run python replay.py with pinned NumPy. Anonymous development scalars only.\n",
        "replay.py": Path(replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={replay.np.__version__}\n".encode(),
    }
    files["manifest.json"] = base.encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, value in sorted(files.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), value, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(out, base.sha(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
