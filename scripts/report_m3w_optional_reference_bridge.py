"""Export aggregate counts without committing per-recording solver dumps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_optional_reference_bridge as study


def main():
    run, root = study.run, study.OUT
    reg, raw, creg, clipped = [root / f"{k}.json" for k in ("registration", "analysis", "clipped_registration", "clipped")]
    r, a, cr, c = [run.read(p) for p in (reg, raw, creg, clipped)]
    if (a["registration"] != run.sha(reg) or c["registration"] != run.sha(creg)
            or cr["parent"]["sha256"] != run.sha(raw) or any(x["confirmation_access"] for x in (r, a, cr, c))):
        raise ValueError("Capacity receipt must match both registrations")
    for d in a["summary"]:
        q = c["summary"][d]
        expected = [a["summary"][d][k]["counts"]["verified_positive_both"] for k in ("frozen_optional", "union")]
        if [q["positive_original"], q["positive_unrestricted"]] != expected:
            raise ValueError("Incompatible original and unrestricted witness counts")
    views = [v["union"]["incumbent"] for rows in a["groups"]["hannover"].values()
             for cell in rows for v in cell["recordings"].values()]
    summary = dict(source_bound=[run.artifact(p) for p in (reg, raw, creg, clipped)], menus=a["summary"],
        radius_comparison=c["summary"], hannover_union_vs_ols4_range_pp=[min(r["gains_pp"][1] for r in views),
                                                                      max(r["gains_pp"][1] for r in views)],
        labels_hindsight_only=True, original_2pct=True, no_training=True, confirmation_access=False)
    run.once(root / "readout.json", run.stable(summary))


if __name__ == "__main__":
    main()
