"""Export compact evidence and a de-identified scalar replay."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_prefix_expert_mixture as study


def build(out):
    r = study.OUT / "registration.json"
    f = study.OUT / "freeze.json"
    a = study.OUT / "analysis.json"
    m = study.OUT / "radius_analysis.json"
    raw, radius = study.read(a), study.read(m)
    if (raw["registration"] != study.sha(r) or raw["freeze"] != study.sha(f)
            or radius["parent"] != study.artifact(a) or raw["confirmation_access"]
            or radius["confirmation_access"] or not raw["original_2pct"]):
        raise ValueError("Require both bound source-development readouts")
    seeds = {s: i + 1 for i, s in enumerate(study.SEEDS)}
    roster = {v["recording"]: v["source"] for v in raw["rows"]}
    src = {s: f"S{i:02d}" for i, s in enumerate(sorted(set(roster.values())))}
    rec = {s: f"V{i:02d}" for i, s in enumerate(sorted(roster))}
    statistics = dict(schema_version=1, seeds=list(seeds.values()),
        roster={rec[k]: src[v] for k, v in roster.items()},
        recording_studies={"prefix_experts": pack.trim(raw["rows"], rec, src, seeds)},
        joint_cells={s: {str(i): {} for i in seeds.values()} for s in src.values()})
    summary = {**raw["summary"], "seeds": list(seeds.values())}
    expected = dict(prefix_experts=summary, joint_pair_contrasts={})
    blob = pack.raw_json(statistics)
    if pack.replay.recompute(statistics) != expected:
        raise ValueError("Anonymous masses do not reproduce the published result")
    files = {name: (study.prior.ROOT / name).read_bytes() for name in pack.FILES}
    files.update({
        "README.md": b"Run python replay.py with pinned NumPy. Anonymous recording-level scalars only.\n",
        "replay.py": Path(pack.replay.__file__).read_bytes(),
        "requirements.txt": f"numpy=={study.np.__version__}\n".encode(),
        "statistics.json": blob,
        "expected.json": pack.raw_json(expected),
    })
    files["manifest.json"] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(out, files)
    study.once(study.OUT / "readout.json", study.stable(dict(
        registration=study.artifact(r), freeze=study.artifact(f), raw_local=study.artifact(a),
        radius=study.artifact(m), anonymous_archive_sha256=study.sha(out), raw=raw["summary"],
        matched=radius["summary"], matched_edits=radius["matched_edits"], raw_nonzero=radius["raw_nonzero"],
        original_2pct=True, confirmation_access=False, deployment_changed=False, submission_ready=False)))
    print(study.sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=study.PRIVATE / "anonymous_prefix_experts.zip")
    build(p.parse_args().out)
