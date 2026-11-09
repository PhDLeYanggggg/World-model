"""Attribute failures of frozen confidence ranks without choosing a policy."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_position_release as run


def ranks(rows, arm):
    groups = {}
    for r in rows:
        groups.setdefault(r["scores"][arm], []).append(r["result"])
    ordered = [groups[s] for s in sorted(groups, reverse=True)]
    if not ordered:
        return dict(views=0, positive=0, ap=None, top1=0., top5=0., top10=0., first_group=[])
    positive = sum(z["eligible"] for g in ordered for z in g)
    tp, seen, ap = 0, 0, 0.
    top = Counter()
    for g in ordered:
        hits = sum(z["eligible"] for z in g)
        for k in (1, 5, 10):
            top[k] += hits*min(len(g), max(k-seen, 0))/len(g)
        tp += hits
        seen += len(g)
        ap += tp/seen*hits
    first = ordered[0]
    return dict(views=seen, positive=positive, ap=ap/positive if positive else None,
                top1=top[1], top5=top[5], top10=top[10], first_group=first)


def main():
    key, _, _, _, parent, previous = run.register()
    p = run.OUT/"summary.json"
    s = run.study.read(p)
    f = run.study.read(run.study.checked(s["freeze"]))
    if f["registration"] != key or f["outcomes_read"] or s["raw_future_opened"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require frozen confidence readout")
    out, verified = {}, {}
    for role in ("calibration", "held"):
        scores = {k: [] for k in run.model.ARMS}
        receipts, replay = {}, []
        for name, ref in f["folds"].items():
            z = run.study.read(run.study.checked(ref))
            rows = list(run.role_rows(z, parent["folds"][name], name, role))
            raw = run.prior.LOCAL/f"{name}.{role}.score.json"
            receipts[name] = run.study.artifact(raw)
            replay.extend(run.study.read(raw)["rows"])
            for k in scores:
                scores[k].append(ranks(rows, k))
        if run.prior.summarize(replay) != previous[role]["arms"]:
            raise ValueError("Original outcome metrics did not replay")
        verified[role] = receipts
        out[role] = {}
        for k, views in scores.items():
            defined = [r for r in views if r["ap"] is not None]
            bad = [r for z in views for r in z["first_group"] if not r["eligible"]]
            out[role][k] = dict(available_folds=len(defined), empty_folds=sum(not r["views"] for r in views),
                all_qualifying_first_group_folds=sum(bool(r["first_group"] and all(x["eligible"] for x in r["first_group"])) for r in views),
                fold_mean_ap=sum(r["ap"] for r in defined)/len(defined) if defined else None,
                fold_mean_base_rate=sum(r["positive"]/r["views"] for r in defined)/len(defined) if defined else None,
                expected_qualified_top1=sum(r["top1"] for r in views),
                expected_qualified_top5=sum(r["top5"] for r in views),
                expected_qualified_top10=sum(r["top10"] for r in views),
                wrong_first_group_views=len(bad), first_group_failed_risk=sum(not r["risk"] for r in bad),
                first_group_failed_gain=sum(r["gain"] is None or min(r["gain"]) <= 0. for r in bad))
    run.base.immutable(run.OUT/"confidence_attribution.json", dict(source=run.study.artifact(p), **out,
        result_source="fresh_run", outcome_source="cached_verified", used_for_selection=False,
        independent_confirmation=False, reserved_outcomes_opened=False))
    run.base.immutable(run.OUT/"outcome_lineage.json", dict(source=run.study.artifact(run.prior.OUT/"summary.json"),
        original_metrics_replayed=True, receipts=verified, raw_future_opened=False, reserved_outcomes_opened=False))
    print(run.study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
