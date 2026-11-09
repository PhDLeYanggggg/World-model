"""Export the fixed-budget timing readout without identifiers or source paths."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from package_m3w_av2_selection_evidence import digest, encode
from plot_m3w_av2_incumbent_timing import DEFAULT, MODES, REPORT, ROOT, numbers, plot


TEMPLATE = ROOT/"reproducibility/av2_incumbent_timing"
FIELDS = ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")
LEVELS = ("local_timing", "same_actor_and_peak", "same_actors", "all_recipients")


def evidence():
    numbers()
    summary, audit = [json.loads((REPORT/k).read_bytes()) for k in ("summary.json", "capacity_attribution.json")]
    if set(summary["by_city"]) != set(audit["by_city"]):
        raise ValueError("Different source rosters")
    groups = []
    for city in sorted(summary["by_city"]):
        c = summary["by_city"][city]
        groups.append(dict(views=c["two_slots"], anchor_eligible=c["anchor_eligible"],
            capacity=audit["by_city"][city], top={m: c[f"{m}_top"] for m in MODES}))
    released = {m: [dict(anchor_eligible=r["anchor_eligible"], risk=[r["risk"][k] for k in FIELDS],
                    relative=r["relative"], gain=r["gain"]) for r in summary["releases"][m]] for m in MODES}
    expected = dict(two_slot_views=summary["counts"]["two_slots"],
        capacity={k: audit["counts"][k] for k in LEVELS},
        top={m: summary["counts"][f"{m}_top"] for m in MODES},
        released={m: summary["counts"][f"{m}_release"] for m in MODES},
        passed={m: summary["counts"][f"{m}_pass"] for m in MODES}, development_only=True)
    return dict(groups=groups, released=released), expected


def package(destination):
    data, expected = evidence()
    with TemporaryDirectory() as tmp:
        p = Path(tmp)
        plot(p/"figure")
        items = {k: (TEMPLATE/k).read_bytes() for k in ("README.md", "verify.py")}
        items.update({"evidence.json": encode(data), "figure.pdf": (p/"figure.pdf").read_bytes()})
        items["manifest.json"] = encode(dict(members={k: digest(v) for k, v in items.items()}))
        for name, raw in items.items():
            if re.search(rb"/Users/|/home/|/cephfs/|github.com|kcl.ac.uk|yangyue|PhDLeYang", raw, re.I):
                raise ValueError("An identifier leaked into the archive")
            (p/name).write_bytes(raw)
        out = subprocess.run([sys.executable, "-I", "-B", "-S", "verify.py"], cwd=p,
                             check=True, capture_output=True, text=True, timeout=20)
        replay = json.loads(out.stdout)
        if replay != expected:
            raise ValueError("Anonymous replay did not match the frozen readout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.zip")
    with ZipFile(pending, "w", compression=ZIP_DEFLATED) as z:
        for name, raw in sorted(items.items()):
            z.writestr(ZipInfo(name, (1980, 1, 1, 0, 0, 0)), raw, compress_type=ZIP_DEFLATED)
    pending.replace(destination)
    print(json.dumps(dict(archive_sha256=digest(destination.read_bytes()), result=replay), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT.with_suffix(".zip"))
    package(p.parse_args().output)
