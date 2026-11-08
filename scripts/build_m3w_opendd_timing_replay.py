"""Package only anonymous scalar evidence for the recent timing experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_candidate_replay as legacy
from src.reproduction import m3w_opendd_timing_replay as replay

OUT = legacy.EXTENDED.parent.parent/"opendd_timing_replay_v1"/"anonymous_opendd_timing.zip"
EXTENDED = OUT.parent.parent/"opendd_timing_replay_v2"/OUT.name
SLATE = "opendd_slate_queue_v1"
RANK = "opendd_slate_rank_v1"
SLATE_ORDER = ("adaptive_queue_vs_cv", "adaptive_queue_vs_same_clock",
               "adaptive_queue_vs_same_credits_immediate", "adaptive_queue_vs_fixed_queue",
               "adaptive_queue_vs_no_geometry", "adaptive_queue_vs_myopic_queue",
               "adaptive_now_vs_fixed_now", "pair_relative")


def read_clock():
    p = legacy.REPORTS/SLATE
    r = json.loads((p/"clock_registration.json").read_text())
    a = json.loads((p/"clock_capacity.json").read_text())
    if (a["registration"] != legacy.sha(p/"clock_registration.json") or not r["preserves_recording_query_counts"]
            or r["parent"]["sha256"] != legacy.sha(p/"analysis.json")
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a))):
        raise ValueError("Require frozen outcome-aware clock diagnostic")
    return a


def build(include_rank=False):
    studies, expected = {}, {}
    names = ("opendd_release_credit_v1", "opendd_credit_value_v2", SLATE)
    for name in (*names, RANK) if include_rank else names:
        a = legacy.read(name)
        s = a["summary"]
        fields = {k: {f: v[f] for f in legacy.FIELDS} for k, v in s["metrics"].items()}
        status = s.get("capacity", {})
        clock = read_clock() if name == SLATE else None
        lookup = {(k, r["seed"]): r["capacity"] for k, rows in clock["by_site"].items() for r in rows} if clock else {}
        scalars = dict(s["effects"])
        if name == SLATE:
            scalars["pair_relative"] = s["pair_relative"]
        cells = []
        for site, rows in sorted(a["by_site"].items()):
            view = []
            for r in sorted(rows, key=lambda v: v["seed"]):
                effect = dict(r["effects"])
                if name == SLATE:
                    effect["pair_relative"] = r["pair_relative"]
                counts = {k: {f: r["metrics"][k][f] for f in v} for k, v in fields.items()}
                for delay, table in status.items():
                    counts[f"capacity_{delay}"] = {k: r["capacity"][delay].get(k, 0) for k in table}
                if clock:
                    counts["same_clock_capacity"] = {k: lookup[site, r["seed"]].get(k, 0) for k in clock["summary"]}
                view.append(dict(effects=effect, counts=counts))
            cells.append(view)
        order = list(SLATE_ORDER if name == SLATE else sorted(scalars))
        if set(order) != set(scalars):
            raise ValueError("Unexpected contrast order")
        studies[name] = dict(sites=cells, effect_order=order, sequential_draws=name == SLATE)
        fields.update({f"capacity_{k}": v for k, v in status.items()})
        if clock:
            fields["same_clock_capacity"] = clock["summary"]
        expected[name] = dict(counts=fields, effects={k: {f: v[f] for f in ("point", "interval95")} for k, v in scalars.items()})
    replay.verify(replay.aggregate(studies), expected)
    files = {
        "replay.py": Path(replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={np.__version__}\n".encode(),
        "README.md": b"# OpenDD timing evidence\n\nRun python replay.py with the pinned NumPy version.\n"
                     b"These anonymous site/seed scalars reproduce summaries and intervals, not training.\n"
                     b"The same-clock optimum and release ceilings are outcome-aware diagnostics.\n"
                     b"None of the results is independent confirmation or deployment evidence.\n",
        "statistics.json": legacy.encoded(dict(studies=studies, independent_confirmation=False, original_risk_cap=.02)),
        "expected.json": legacy.encoded(expected),
    }
    files["manifest.json"] = legacy.encoded({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    out = EXTENDED if include_rank else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pending")
    with ZipFile(tmp, "w") as z:
        for k, v in sorted(files.items()):
            item = ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            z.writestr(item, v, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(out)
    print(out, legacy.sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-rank", action="store_true")
    build(p.parse_args().include_rank)
