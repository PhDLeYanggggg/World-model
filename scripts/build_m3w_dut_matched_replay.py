"""Build a deterministic anonymous DUT scalar packet without raw data or weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_m3w_opendd_candidate_replay import ROOT, REPORTS, encoded, sha
from src.reproduction import m3w_dut_matched_replay as replay

OUT = REPORTS / "dut_matched_motion_v1"
DEFAULT = ROOT / "data/stage_cvpr2027_experiments/dut_matched_motion_v1/anonymous_dut_motion.zip"
CV_ALIAS = "constant_velocity_causal_fd"


def inputs():
    a = json.loads((OUT / "analysis.json").read_text())
    r = OUT / "registration.json"
    reg = json.loads(r.read_text())
    if (a["registration"] != sha(r) or not a["original_2pct"] or a["confirmation_access"]
            or a["training"] or not reg["no_refit"] or not reg["no_target_tuning"]):
        raise ValueError("Require frozen exposed DUT evaluation")
    for k in ("code", "method", "source", "prior"):
        if sha(ROOT / reg[k]["path"]) != reg[k]["sha256"]:
            raise ValueError("Changed lineage")
    return a


def collect():
    a = inputs()["summary"]

    def cell(c):
        arms = {}
        for k, v in c["arms"].items():
            if k == CV_ALIAS:
                continue
            s = v["slices"]
            arms[k] = dict(ade=s["complete"]["ade"], fde=s["endpoint"]["fde"], easy=s["easy"]["delta_pct"],
                           **{f: v[f] for f in ("lower_pp", "edits", "undefined_selected_easy")})
        return dict(arms=arms, **{f: c[f] for f in ("paired", "strict", "common")})

    sites = [[cell(v) for v in a["site"][k].values()] for k in sorted(a["site"])]
    expected = {k: v for k, v in a.items() if k != "site"}
    expected["arms"] = {k: v for k, v in a["arms"].items() if k != CV_ALIAS}
    replay.verify(replay.summarize(sites), expected)
    return sites, expected


def build(out):
    sites, expected = collect()
    files = {"README.md": b"Run python replay.py with pinned NumPy. Two-site exposed DUT scalars only.\n",
             "requirements.txt": f"numpy=={replay.np.__version__}\n".encode(),
             "replay.py": Path(replay.__file__).read_bytes(),
             "scalars.json": encoded(sites), "expected.json": encoded(expected)}
    files["manifest.json"] = encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, v in sorted(files.items()):
            z.writestr(ZipInfo(k, (1980, 1, 1, 0, 0, 0)), v, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    build(p.parse_args().output)
