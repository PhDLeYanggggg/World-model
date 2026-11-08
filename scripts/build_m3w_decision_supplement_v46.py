"""Add fixed-budget timing and direction controls to the working supplement."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v45 as previous
from scripts import build_m3w_decision_supplement_v20 as package

PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v46"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
BASE = previous.BASE


def build(out, style, archive):
    timing = PUBLIC.parent / "query_temporal_envelope_v1"
    direction = PUBLIC.parent / "matched_direction_v2"
    files = [timing / "analysis.json", timing / "capacity.json", direction / "analysis.json",
             direction / "recipient_capacity_registration.json", direction / "recipient_capacity.json",
             direction / "recipient_capacity_replay.json"]
    t, peak, a, registration, c, replay = [json.loads(p.read_text()) for p in files]
    digest = BASE.cal.digest
    if (any(d["confirmation_access"] for d in (t, peak, a, registration, c, replay))
            or peak["analysis"]["sha256"] != digest(files[0])
            or registration["direction"]["sha256"] != digest(files[2])
            or c["registration"] != digest(files[3]) or not c["same_query_edit_counts"]
            or replay["parent"]["sha256"] != digest(files[4])
            or not a["hindsight_never_used_for_selection"] or a["fits"] != 54):
        raise ValueError("Require the bound matched-direction evidence")
    def contrast(data, label):
        effect = data["summary"]["effects"][label]["completion_lower_pp"]
        lo, hi = effect["interval95"]
        return f'{effect["point"]:+.3f} pp [{lo:+.3f}, {hi:+.3f}]'
    risk = a["summary"]["risk"]
    if [risk[k]["switches"] for k in ("flat", "ego", "neighbors")] != [95769]*3:
        raise ValueError("The matched comparison must retain intervention count")
    values = dict(Timing=contrast(t, "learned_vs_flat"), Ego=contrast(a, "ego_vs_flat"),
        Peer=contrast(a, "neighbors_vs_ego"), TimingPass=peak["summary"]["risk"]["oracle"]["supported_views"],
        DirectionPass=a["hindsight"]["risk"]["oracle"]["supported_views"],
        FlatBoth=c["summary"]["quarter"]["verified_both"],
        CircleBoth=replay["circle_summary"]["verified_both"],
        FlatUnresolved=c["summary"]["quarter"]["unresolved"],
        CircleUnresolved=replay["circle_summary"]["unresolved"])
    doc = previous.build(out, style, archive, "supplement")
    tex, defs = out / "supplement.tex", out / "supplement_values.tex"
    frag = PUBLIC / "supplement_matched_geometry.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text() + "\n\n" + marker))
    defs.write_text(defs.read_text() + "\n".join(
        rf"\newcommand{{\MatchedGeo{k}}}{{{v}}}" for k, v in values.items()) + "\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in (*files, frag)})
    for p in (tex, defs):
        doc["files"][p.name] = digest(p)
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=DEFAULT / "supplement")
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    cli.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    args = cli.parse_args()
    d = build(args.output_dir, args.style, args.archive)
    if args.source_zip:
        package.package(args.output_dir, args.source_zip, d)
