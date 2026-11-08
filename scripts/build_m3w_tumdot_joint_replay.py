"""Export anonymous TUMDOT mask attribution and joint-teacher scalars."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_tumdot_selector_replay as prior

ROOT, REPORTS, replay = prior.ROOT, prior.REPORTS, prior.replay
NAMES = ("tumdot_three_reference_v1", "tumdot_joint_teacher_v1")
DEFAULT = ROOT/"data/stage_cvpr2027_experiments/tumdot_joint_replay_v1/anonymous_joint.zip"


def inputs():
    prior.inputs()
    result = []
    for name in NAMES:
        folder = REPORTS/name
        reg, f = (folder/f"{k}.json" for k in ("registration", "analysis"))
        a, r = (json.loads(p.read_text()) for p in (f, reg))
        if (a["registration"] != prior.sha(reg) or not a["original_2pct"] or a["confirmation_access"]
                or not a["source_is_previously_exposed"]):
            raise ValueError("Require frozen development scores")
        if "heads" in a and a["heads"] != prior.sha(folder/"head_freeze.json"):
            raise ValueError("Changed checkpoint receipt")
        for z in [r["code"], r["model"], *r["parents"]]:
            if prior.sha(ROOT/z["path"]) != z["sha256"]:
                raise ValueError("Changed TUMDOT lineage")
        result.append(a)
    return result


def attribution_row(z):
    m, v = z["decomposition"], z["oracle"]
    counts = {f"{k}_{s}": [d["rows"], d["changed"], 0] for k, q in m.items() for s, d in q.items()}
    counts["oracle"] = [1, int(v["passes"]), int(v["status"] == "optimal")]
    effects = {f"{k}_{s}_{f}": d[f] for k, q in m.items() for s, d in q.items()
               for f in ("observed", "debit", "lower")}
    return dict(counts=counts, effects=effects)


def collect():
    studies, expected = {}, {}
    for label, a in zip(("audit", "fit"), inputs(), strict=True):
        r = a["rows"]
        sites = sorted({k.split("-")[0] for k in r})
        for part, summary in a["summary"].items():
            key = f"{label}_{part}"
            make = attribution_row if label == "audit" else prior.row
            studies[key] = [[make(z["scores"][part]) for k, group in sorted(r.items())
                             if k.startswith(s+"-") for z in group] for s in sites]
            if label == "audit":
                m = summary["decomposition"]
                counts = {f"{k}_{s}": [d["rows"], d["changed"], 0] for k, q in m.items() for s, d in q.items()}
                v = summary["oracle"]
                counts["oracle"] = [v["views"], v["joint"], v["statuses"].get("optimal", 0)]
                e = {f"{k}_{s}_{f}": d[f] for k, q in m.items() for s, d in q.items()
                     for f in ("observed", "debit", "lower")}
            else:
                m = summary["metrics"]
                counts = {k: [d[f] for f in ("risk_pass", "joint_cv", "joint_cv_ols")] for k, d in m.items()}
                e = {f"{k}_{n}": d[n] for k, d in summary["effects"].items() for n in ("lower", "observed")}
                e.update({f"{k}_{n}": d[n] for k, d in m.items() for n in ("cv", "ols4")})
            expected[key] = dict(metrics=counts, effects={k: [d["point"], *d["interval95"]] for k, d in e.items()})
    return studies, expected


def build(out):
    studies, expected = collect()
    replay.verify(replay.aggregate(studies), expected)
    files = {
        "scalars.json": prior.encoded(studies), "expected.json": prior.encoded(expected),
        "README.md": b"Run python replay.py with pinned NumPy. Anonymous development scalars only.\n",
        "replay.py": Path(replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={replay.np.__version__}\n".encode(),
    }
    files["manifest.json"] = prior.encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, value in sorted(files.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), value, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(out, prior.sha(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
