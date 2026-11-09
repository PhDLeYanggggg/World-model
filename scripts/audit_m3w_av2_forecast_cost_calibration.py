"""Separate candidate availability, additive eligibility and ranking on calibration."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_forecast_cost as run

np, torch, study, base, old, prior, eq, prev = (
    run.np, run.torch, run.study, run.base, run.old, run.prior, run.eq, run.prev)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, source, roles, _, outer, control, _, _ = run.register()
    report_path = run.OUT/"summary.json"
    report = study.read(report_path)
    freeze = study.read(study.checked(report["freeze"]))
    if freeze["registration"] != key or freeze["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("The experiment must be sealed before the audit")
    by_fold, totals, clips = {}, Counter(), set()
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        cuts = {city: prev.donor_cut(data, {n: t for n, t in dy.items() if data[n]["city"] != city}).tolist()
                for city in donors}
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold = freeze["folds"][name]
            cut = fold["identity"]["cut"]
            for city, ref in fold["references"].items():
                ck = torch.load(study.checked(ref["checkpoint"]), weights_only=True, map_location="cpu")
                ident = dict(fold["identity"], recipient=city, donors=[s for s in donors if s != city], action_cut=cuts[city])
                if ck["identity"] != ident or ck["step"] != run.STEPS:
                    raise ValueError("Nested checkpoint or donor-only cutoff failed replay")
            action = prior.load_action(control["folds"][name])
            raw, offsets = eq.predict(old.load_outer(outer["folds"][name]), data, list(cy))
            counts = Counter()
            for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
                d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
                prepared = run.slate(d, seed, cut, action)
                if prepared is None:
                    continue
                paths, a = prepared[:2]
                _, q, _ = prev.label_terms(d, paths, a, *cy[n], cut)
                add = eq.ceiling.eligible_additive(prev.metrics(a, d, *cy[n], cut))
                counts.update(two_slots=1, additive_eligible=int(add), joint_available=int(q.any()),
                    paired_available=int(add and q.any()), qualified_options=int(q.sum()),
                    paired_qualified_options=int(q.sum()) if add else 0)
                if add and q.any():
                    clips.add(n)
            for m, val in fold["calibration"].items():
                if not (val["accepted"] <= val["strict_with_add"] <= counts["paired_available"]):
                    raise ValueError("Observed decisions exceed exact calibration capacity")
                counts.update({f"{m}_chosen_both": val["strict_with_add"], f"{m}_released": val["accepted"]})
            by_fold[name] = dict(counts)
            totals.update(counts)
    out = dict(result_source="fresh_run", input=study.artifact(report_path), code=study.artifact(Path(__file__)),
        by_fold=by_fold, totals=dict(totals), unique_paired_scenarios=len(clips),
        verified_inner_checkpoint_cutoffs=report["new_nested_references"],
        utility_scoring_calibration_only=True, hindsight_oracle_not_deployable=True, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"calibration_capacity.json", out)
    print(study.json.dumps(out["totals"], indent=2))


if __name__ == "__main__":
    main()
