"""Audit the fixed-path easy gate's released cases and remaining bottleneck."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_easy_gate as run

np, torch, study, base, av, prev, eq, old = run.np, run.torch, run.study, run.base, run.av, run.prev, run.eq, run.old


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, outer, _, parent, previous, _ = run.register()
    _, _, _, _, _, _, control, _ = run.prior.register()
    p = run.OUT/"summary.json"
    report = study.read(p)
    seal = study.read(study.checked(report["freeze"]))
    if seal["registration"] != key or seal["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Audit only the sealed development result")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows, cases = [], {k: [] for k in run.MODES}
    tally, by_city = Counter(), {}
    for name, f in seal["folds"].items():
        city, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        bucket = by_city.setdefault(city, Counter())
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(parent["folds"][name]["forecast"]), allow_pickle=False) as path,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a,
              np.load(study.checked(previous["folds"][name]["forecast"]), allow_pickle=False) as prior):
            for bank in (path, raw, a, prior):
                for k in ("names", "offsets"):
                    np.testing.assert_array_equal(z[k], bank[k])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                j = int(path["ix_upper"][i])
                if j < 0:
                    continue
                d, (y, v) = data[n], truth[n]
                if d["city"] != city:
                    raise ValueError("Wrong held city")
                chosen = path["upper"][lo:hi]
                m = old.action.menu(dict(d, bank=eq.expanded(d, raw["raw_eq"][lo:hi])), seed, cut)
                target, metrics = run.target(d, m, j, chosen, y, v, cut)
                risk = metrics["risk"]
                support, easy = target[:2].bool().tolist()
                risk_all = risk["completion_all_risk_upper"]
                all_ok = risk_all is not None and risk_all <= .02
                dual = metrics["gain"] is not None and min(metrics["gain"]) > 0.
                ok = eq.ceiling.eligible_additive(metrics)
                stats = dict(two_slots=1, easy_support=int(support), easy_pass=int(easy), eligible=int(ok),
                    easy_pass_all_fail=int(easy and not all_ok),
                    easy_pass_all_ok_dual_fail=int(easy and all_ok and not dual))
                if ok != bool(easy and all_ok and dual):
                    raise ValueError("Risk decomposition must reproduce qualification")
                tally.update(stats)
                bucket.update(stats)
                if support:
                    y_term = target[2:].numpy()
                    delta = (y_term <= z["tail"][i]).astype(int)-(y_term <= prior["harm_selected"][i, 1:]).astype(int)
                    rows.append(dict(city=city, clip=str(n), seed=seed, delta=delta.tolist()))
                active = [k for k in run.MODES if (t := f["calibration"][k]["threshold"]) is not None and z[f"score_{k}"][i] >= t]
                if active:
                    anchor = a["additive"][lo:hi]
                    fixed = prev.metrics(anchor, d, y, v, cut)
                    gain = run.prior.cost.audit.direct_trajectory(chosen, anchor, d["cv"], y, v)
                    pair = float(av.frontier.pair_reduction(chosen[None], anchor, y, v)[0, 1])
                    x = dict(fold=name, eligible=ok, easy_support=support, easy_pass=easy,
                        all_pass=all_ok, dual_gain=dual, risk={k: risk[k] for k in
                            ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")},
                        fixed_eligible=eq.ceiling.eligible_additive(fixed), trajectory_lower_vs_fixed_pp=gain[2] if gain else None,
                        pair_lower_vs_fixed=pair)
                    for k in active:
                        cases[k].append(x)
    if any(tally[k] != report["counts"][k] for k in ("two_slots", "easy_support", "easy_pass", "eligible")):
        raise ValueError("Held event totals changed")
    if any(len(r) != report["counts"][f"{k}_released"] for k, r in cases.items()):
        raise ValueError("Held releases did not replay")
    out = dict(source=study.artifact(p), code=study.artifact(Path(__file__)), result_source="fresh_run",
        by_city={k: dict(v) for k, v in by_city.items()}, bottleneck=dict(tally), releases=cases,
        supported_tail_coverage_delta=av.interval(rows, lambda r: r["delta"]),
        audit_only=True, reserved_outcomes_opened=False, forecasts_or_thresholds_changed=False)
    base.immutable(run.OUT/"release_audit.json", out)
    print(study.json.dumps({k: out[k] for k in ("bottleneck", "releases", "supported_tail_coverage_delta")}, indent=2))


if __name__ == "__main__":
    main()
