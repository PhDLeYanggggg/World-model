"""Verify matched amplitudes and summarize actual risk failures from frozen scores."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dense_motion_calibration as base
from scripts import run_m3w_dense_safe_union as union


run, np = base.run, base.np


def main():
    registration, _, _, parent, roster, _, _ = base.register()
    a = run.read(base.PUBLIC/"analysis.json")
    b = run.read(union.OUT/"analysis.json")
    if a["registration"] != registration or b["registration"] != run.sha(union.OUT/"registration.json"):
        raise ValueError("Frozen analyses must bind their registrations")
    ready = run.read(union.OUT/"forecast_freeze.json")
    rows = []
    matched_queries = 0
    for entry, clip in zip(ready["clips"], parent["clips"], strict=True):
        rec = entry["recording"]
        if rec != clip["sequence"] or rec not in roster:
            raise ValueError("Unexpected recording")
        r = run.read(union.PRIVATE/f"score_{rec}.json")
        if r["choices"] != entry["choices"] or r["registration"] != b["registration"]:
            raise ValueError("Scoring checkpoint mismatch")
        rows.extend(r["rows"])
        with np.load(run.checked(clip["artifact"]), allow_pickle=False) as z:
            offsets = z["offsets"][:-1]
        with np.load(run.checked(entry["choices"]), allow_pickle=False) as z:
            choices = z["choices"]
        for seed in choices:
            for level in range(5):
                counts = np.add.reduceat(seed == level, offsets, axis=1, dtype=np.int64)
                np.testing.assert_array_equal(counts[0], counts[1])
            matched_queries += len(offsets)
    if base.grouped_readout(rows, roster, base.cal.SEEDS) != b["summary"]:
        raise ValueError("Union aggregate did not replay")
    risk = {arm:[r["metrics"][arm]["risk"] for r in rows] for arm in ("union","matched")}
    names = ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")
    failures = {k:{n:sum(v[n] is not None and v[n] > .02 for v in r) for n in names}
                for k,r in risk.items()}
    transitions = [int(x["supported"])-int(y["supported"])
                   for x,y in zip(risk["union"], risk["matched"], strict=True)]
    doc = dict(
        dense_analysis_sha256=run.sha(base.PUBLIC/"analysis.json"),
        union_analysis_sha256=run.sha(union.OUT/"analysis.json"),
        same_query_amplitude_verified=matched_queries, actual_failures=failures,
        union_lost_passes=transitions.count(-1), union_gained_passes=transitions.count(1),
        actual_union_worst_all_ratio=max(r["completion_all_risk_upper"] for r in risk["union"]),
        actual_union_worst_easy_ratio=max(r["completion_easy_risk_upper"] for r in risk["union"]),
        predicted_union_worst_all_ratio=max(r["predicted_risks"][0]
            for clip in ready["clips"] for r in clip["diagnostics"]),
        predicted_union_worst_easy_ratio=max(r["predicted_risks"][1]
            for clip in ready["clips"] for r in clip["diagnostics"]),
        future_labels_reopened=False, confirmation_access=False,
    )
    run.once(union.OUT/"verification.json", run.stable(doc))
    print(json.dumps(doc))


if __name__ == "__main__":
    main()
