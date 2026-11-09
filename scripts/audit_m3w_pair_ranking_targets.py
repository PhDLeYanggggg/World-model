"""Check whether local feasible slates can also improve the matched pair objective."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_pair_ranking as rank

pair, np, study, base = rank.pair, rank.np, rank.study, rank.base
frontier, lower = pair.audit.frontier, pair.audit.lower


def targets(p, paths, ref, y, valid, cut):
    local = pair.labels(p, paths, y, valid, cut)[:, 0] > 0.
    paired = frontier.pair_reduction(paths, ref, y, valid)[:, 1] > 1e-10
    accuracy = np.asarray([lower(ref, path, y, valid).sum() for path in paths]) >= -1e-10
    return local, local & paired & accuracy


def audit(inputs, truth, cut, predictions):
    totals = Counter()
    for ix, p in pair.menus(inputs):
        ref = predictions["forced_add"][ix]
        y, valid = truth["y"][ix], truth["v"][ix]
        local, both = False, False
        for first in range(0, len(p["opts"]), 128):
            paths = pair.paths_for(p, p["opts"][first:first+128])
            a, b = targets(p, paths, ref, y, valid, cut)
            local, both = local or a.any(), both or b.any()
            if both:
                break
        top, top_both = targets(p, predictions["forced_joint"][ix][None], ref, y, valid, cut)
        totals.update(queries=1, feasible=int(local), feasible_and_pair=int(both),
                      selected_feasible=int(top[0]), selected_feasible_and_pair=int(top_both[0]))
    return {k: totals[k] for k in ("queries", "feasible", "feasible_and_pair", "selected_feasible",
                                  "selected_feasible_and_pair")}


def main():
    report_path = rank.OUT/"summary.json"
    report = study.read(report_path)
    f = study.read(study.checked(report["freeze"]))
    _, cohorts, _ = rank.register()
    if f["registration"] != report["registration"] or f["held_outcomes_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Require the sealed development comparison")
    data = pair.run.turn.exp.loader(cohorts)
    inputs = {s: pair.causal(cohorts[s]) for s in rank.SITES}
    rows = {}
    for name, item in f["folds"].items():
        ident = item["fit"]["identity"]
        if ident["method"] != "matched_rank":
            continue
        cal = ident["cal"]
        v = audit(inputs[cal], data[cal], item["cut"][0], rank.checked_npz(item["cal"]))
        parent_name = f"{ident['held']}_{ident['seed']}"
        before = report["calibration_capture"][parent_name]
        if v["feasible"] != before["opportunities"] or v["selected_feasible"] != before["matched_rank"]:
            raise ValueError("Local feasibility counts did not replay")
        rows[parent_name] = dict(site=cal, **v)
    result = dict(summary=study.artifact(report_path), result_source="fresh_run",
                  calibration_only=True, no_threshold_changes=True, rows=rows,
                  totals=dict(sum((Counter({k: v for k, v in r.items() if k != "site"}) for r in rows.values()), Counter())),
                  independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(rank.OUT/"target_audit.json", result)
    print(study.json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
