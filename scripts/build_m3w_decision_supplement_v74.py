"""Bind the peer falsification and expiry decomposition to the supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v73 as previous

BASE = previous.BASE
PUBLIC = previous.PUBLIC.parent / "evidence_manuscript_v74"
DEFAULT = previous.DEFAULT.parent / PUBLIC.name / "supplement"


def build(out, style, archive):
    src = PUBLIC.parent / "wait_peer_value_v1"
    paths = [src / f"{k}.json" for k in ("registration", "analysis", "head_freeze", "expiry/registration", "expiry/analysis")]
    r, a, h, er, e = [json.loads(p.read_text()) for p in paths]
    digest = BASE.cal.digest
    if (any(x["registration"] != digest(paths[0]) for x in (a, h)) or a["heads"] != digest(paths[2])
            or e["registration"] != digest(paths[3]) or er["parents"][0]["sha256"] != digest(paths[1])
            or any(x["confirmation_access"] or not x["original_2pct"] for x in (r, a, er, e))):
        raise ValueError("Require the hash-bound peer ablation and expiry replay")
    q, v = a["summary"], e["summary"]["peer"]
    u, m = (q["direct"][k] for k in ("peer_vs_anchor", "peer_vs_emission_match"))
    if abs(u["point"]-m["point"]-v["macro_gain_pp"]) > 1e-10:
        raise ValueError("Expected the exact single-toggle expiry decomposition")

    def effect(z, unit="", precision=4):
        lo, hi = z["interval95"]
        return f'{z["point"]:+.{precision}f}{unit} [{lo:+.{precision}f}, {hi:+.{precision}f}]'

    c = q["joint_ap_contrast"]
    values = dict(Offers=f'{q["opportunities"]["n"]:,}', Pos=q["opportunities"]["positive"], Locations=c["self"]["locations"],
        SelfAP=effect(c["self"]), WrongAP=effect(c["wrong_peer"]), Joint=q["arms"]["peer"]["dual_positive"],
        Anchor=q["arms"]["anchor"]["dual_positive"], Full=effect(u, " pp"), Match=effect(m, " pp", 5),
        Expired=q["actions"]["peer"]["expired"], Proxy=v["expiries"]["no_positive_prefix_candidate"],
        Blocked=v["expiries"]["destination_blocked_by_reserve_cap_or_anchor"])
    doc = previous.build(out, style, archive, "supplement")
    tex, defs, frag = out / "supplement.tex", out / "supplement_values.tex", PUBLIC / "peer_wait.tex"
    marker = r"\section*{LLM Usage Statement}"
    tex.write_text(BASE.replace_once(tex.read_text(), marker, frag.read_text()+"\n\n"+marker))
    defs.write_text(defs.read_text()+"\n".join(rf"\newcommand{{\Peer{k}}}{{{v}}}" for k, v in values.items())+"\n")
    doc["source_bindings"].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in (*paths, frag)})
    doc["files"].update({p.name: digest(p) for p in (tex, defs)})
    (out / "supplement_bindings.json").write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    p.add_argument("--source-zip", type=Path)
    p.add_argument("--style", type=Path, default=BASE.v22.STYLE)
    p.add_argument("--archive", type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    doc = build(a.output_dir, a.style, a.archive)
    if a.source_zip:
        previous.previous.previous.supplement_package.package(a.output_dir, a.source_zip, doc)
