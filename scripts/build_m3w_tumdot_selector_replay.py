"""Pack anonymous scalar TUMDOT comparisons; no trajectories or weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_m3w_opendd_candidate_replay import REPORTS, ROOT, encoded, sha
from src.reproduction import m3w_tumdot_selector_replay as replay

NAMES = ("tumdot_matched_selector_v1", "tumdot_incumbent_selector_v1")
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/tumdot_selector_replay_v1/anonymous_tumdot.zip"


def inputs():
    result = []
    for name in NAMES:
        folder = REPORTS/name
        p, q, h = (folder/f"{k}.json" for k in ("analysis", "registration", "head_freeze"))
        a, r, f = (json.loads(x.read_text()) for x in (p, q, h))
        if (a["registration"] != sha(q) or a["heads"] != sha(h) or f["registration"] != a["registration"]
                or not a["original_2pct"] or a["confirmation_access"] or not a["source_is_previously_exposed"]):
            raise ValueError("Require frozen development scores")
        for z in [r["code"], r["model"], *r["parents"]]:
            if sha(ROOT/z["path"]) != z["sha256"]:
                raise ValueError("Changed TUMDOT lineage")
        result.append(a)
    return result


def row(scores):
    m, e = (scores[k] for k in ("metrics", "contrasts"))
    counts = {}
    effects = {f"{k}_{n}": f[n] for k, f in e.items() for n in ("lower", "observed")}
    for k, z in m.items():
        c, o, safe = z["lower_pp"]["cv"], z["lower_pp"]["ols4"], z["risk"]["supported"]
        counts[k] = [int(safe), int(safe and c > 0.), int(safe and min(c, o) > 0.)]
        effects.update({k+"_cv": c, k+"_ols4": o})
    return dict(counts=counts, effects=effects)


def build(out):
    studies, expected = {}, {}
    for name, a in zip(("cost", "pair"), inputs(), strict=True):
        r = a["rows"]
        sites = sorted({k.split("-")[0] for k in r})
        for part, v in a["summary"].items():
            key = f"{name}_{part}"
            studies[key] = [[row(z["scores"][part]) for k, group in sorted(r.items())
                             if k.startswith(s+"-") for z in group] for s in sites]
            counts = {k: [m[f] for f in ("risk_pass", "joint_cv", "joint_cv_ols")] for k, m in v["metrics"].items()}
            effects = {f"{k}_{n}": f[n] for k, f in v["effects"].items() for n in ("lower", "observed")}
            effects.update({f"{k}_{n}": m[n] for k, m in v["metrics"].items() for n in ("cv", "ols4")})
            expected[key] = dict(metrics=counts,
                                 effects={k: [f["point"], *f["interval95"]] for k, f in effects.items()})
    replay.verify(replay.aggregate(studies), expected)
    files = {
        "scalars.json": encoded(studies), "expected.json": encoded(expected),
        "README.md": b"Run python replay.py with the pinned NumPy version. The archive "
                     b"contains anonymous scalar development comparisons, not independent confirmation.\n",
        "replay.py": Path(replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={replay.np.__version__}\n".encode(),
    }
    files["manifest.json"] = encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, value in sorted(files.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), value, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(out, sha(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
