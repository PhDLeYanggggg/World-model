"""Separate calibration-count transport from the frozen AV2 ranking ceiling."""

from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_separate_support_gate as run

study, base, av, np = run.study, run.base, run.av, run.np
OUT = run.OUT.parent/"av2_release_transport_v1"


def prefix(rows, arm, check):
    """Maximum all-passing threshold prefix; ties cannot be separated."""
    groups = defaultdict(list)
    for row in rows:
        score = row["scores"][arm]
        if score is not None:
            if not isfinite(score):
                raise ValueError("Expected finite confidence")
            groups[score].append(row)
    out = []
    for score in sorted(groups, reverse=True):
        group = groups[score]
        if any(not check(row) for row in group):
            return out, group
        out.extend(group)
    return out, []


def eligible(row):
    return row["result"]["eligible"]


def risk(row):
    return row["result"]["risk"]


def counts(rows):
    out = Counter(n=len(rows), qualified=sum(map(eligible, rows)),
                  risk_fail=sum(not risk(r) for r in rows))
    for r in rows:
        g = r["result"]["gain"]
        floor_fail = g is None or min(g) <= 0.
        out.update(floor_fail=int(floor_fail), both_fail=int(floor_fail and not risk(r)))
        if "cause" in r and not risk(r):
            out.update({r["cause"]: 1})
    return dict(out)


def count_match(rows, arm, n):
    if not 0 <= n <= len(rows):
        raise ValueError("Matched count exceeds admissible rows")
    fields = ("n", "qualified", "risk_fail", "floor_fail", "both_fail",
              "unsupported_reference", "observed_positive_harm")
    w = run.weights(np.asarray([r["scores"][arm] for r in rows]), n)
    picked = [(float(v), counts([r])) for r, v in zip(rows, w, strict=True) if v]
    return {k: sum(v*q.get(k, 0) for v, q in picked) for k in fields}


def comparison(cal, held, arm, rule):
    if run.calibrate(cal, arm) != rule:
        raise ValueError("Calibration did not replay")
    t = rule["threshold"]
    candidates = [r for r in held if r["scores"][arm] is not None]
    actual = [r for r in candidates if t is not None and r["scores"][arm] >= t]
    supported = [r for r in candidates if r["support"]]
    strict, first = prefix(candidates, arm, eligible)
    known_strict, known_first = prefix(supported, arm, eligible)
    known_risk, _ = prefix(supported, arm, risk)
    all_q = sum(map(eligible, held))
    admitted_q = sum(map(eligible, candidates))
    matched_n = min(rule["accepted"], len(candidates))
    released, safe = {r["clip"] for r in actual}, {r["clip"] for r in known_strict}
    reachable = released | safe
    return dict(available=dict(held=len(held), admitted=len(candidates), supported=len(supported),
                               qualified=all_q, qualified_admitted=admitted_q),
                calibration_prefix=rule["accepted"],
                actual=counts(actual), matched=count_match(candidates, arm, matched_n),
                retrospective_strict=len(strict), retrospective_risk=len(prefix(candidates, arm, risk)[0]),
                retrospective_known_support_strict=len(known_strict), retrospective_known_support_risk=len(known_risk),
                first_tie=counts(first), first_supported_tie=counts(known_first),
                matched_oracle=min(admitted_q, matched_n),
                misses=dict(gate_excluded=all_q-admitted_q,
                            below_safe_prefix=sum(eligible(r) and r["clip"] not in reachable for r in candidates),
                            inside_safe_prefix=len(safe-released)))


def main():
    run.torch.set_num_threads(4)
    run.torch.set_num_interop_threads(1)
    key, parent, _, _, _ = run.register()
    p = run.OUT/"summary.json"
    summary = study.read(p)
    f, c = [study.read(study.checked(summary[k])) for k in ("freeze", "calibration")]
    if (f["registration"] != key or f["outcomes_read"] or c["freeze"] != summary["freeze"]
            or c["held_receipts_used"] or summary["reserved_outcomes_opened"]
            or summary["independent_confirmation"] or not summary["original_2pct"]
            or set(f["folds"]) != set(c["rules"]) or set(f["folds"]) != set(parent["folds"])):
        raise ValueError("Require verified development-only calibration")
    cells, lineage, totals, clips = {}, {}, {}, set()
    rows_by_arm = {a: [] for a in run.ARMS}
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        pf = parent["folds"][name]
        role = study.read(study.checked(pf))["identity"]
        if (z["identity"]["source"] != pf or z["identity"]["registration"] != key
                or len({role["held"], role["calibration"], *role["donors"]}) != 6):
            raise ValueError("Overlapping source roles or wrong freeze")
        path = run.old.support.LOCAL/f"{name}.support.json"
        ann = study.read(path)
        if ann["parent"] != pf:
            raise ValueError("Wrong support receipt")
        lineage[name] = dict(support=study.artifact(path))
        data = {}
        for side in ("calibration", "held"):
            rows = list(run.old.support.prior.role_rows(z, pf, name, side))
            city = role["held" if side == "held" else "calibration"]
            if (len({r["clip"] for r in rows}) != len(rows)
                    or any(r["city"] != city or r["seed"] != role["seed"] for r in rows)):
                raise ValueError("Wrong city/seed receipt")
            data[side] = rows
            raw = run.old.support.prior.prior.LOCAL/f"{name}.{side}.score.json"
            lineage[name][side] = study.artifact(raw)
        cal, held = data["calibration"], data["held"]
        for r, b in zip(held, ann["cells"], strict=True):
            if r["clip"] != b["clip"] or eligible(r) and (not risk(r) or not b["support"]):
                raise ValueError("Wrong support annotation")
            r.update(support=b["support"], cause=b["cause"])
            clips.add(r["clip"])
        fold = {a: comparison(cal, held, a, c["rules"][name][a]) for a in run.ARMS}
        cells[name] = dict(held_city=role["held"], calibration_city=role["calibration"], arms=fold)
        for a, r in fold.items():
            q = r["available"]["qualified"]
            if r["actual"]["qualified"] + sum(r["misses"].values()) != q:
                raise ValueError("Non-exhaustive missed-qualification decomposition")
            rows_by_arm[a].append(dict(city=role["held"], clip="fold", **r))
    for a, rows in rows_by_arm.items():
        keys = ("calibration_prefix", "matched_oracle", "retrospective_strict", "retrospective_risk",
                "retrospective_known_support_strict", "retrospective_known_support_risk")
        flat = {k: sum(r[k] for r in rows) for k in keys}
        for k in ("available", "actual", "matched", "misses", "first_tie", "first_supported_tie"):
            keys = set().union(*(r[k] for r in rows))
            flat[k] = {m: sum(r[k].get(m, 0) for r in rows) for m in sorted(keys)}
        flat["folds_with_calibration_release"] = sum(r["calibration_prefix"] > 0 for r in rows)
        flat["folds_with_safe_prefix"] = sum(r["retrospective_strict"] > 0 for r in rows)
        flat["folds_with_known_support_safe_prefix"] = sum(r["retrospective_known_support_strict"] > 0 for r in rows)
        flat["matched_shortfall_pp_per_held_view"] = av.interval(rows, lambda r:
            100*(r["matched_oracle"]-r["matched"]["qualified"])/r["available"]["held"])
        prior = summary["held"][a]
        if any(flat["actual"][k] != prior[v] for k, v in (
                ("n", "released"), ("qualified", "qualified"), ("risk_fail", "risk_fail"), ("floor_fail", "floor_fail"))):
            raise ValueError("Actual policy outcomes changed")
        totals[a] = flat
    position = study.read(run.old.support.prior.OUT/"summary.json")
    if (len(clips) != summary["scenarios"] or totals["conditional_harm"]["available"]["held"] != summary["views"]
            or totals["conditional_harm"]["available"]["qualified"] != position["held_forced_eligible"]):
        raise ValueError("Wrong held counterfactual capacity")
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(input=study.artifact(p), code=study.artifact(Path(__file__)), receipt_hashes=lineage,
               totals=totals, by_fold=cells, unique_held_scenarios=len(clips),
               matched_budget="repeat calibration count within each held fold, capped by admitted held availability",
               retrospective_uses_held_labels=True, used_for_selection=False,
               original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False,
               result_source="fresh_run", outcome_source="cached_verified", deployment_changed=False)
    base.immutable(OUT/"transport_audit.json", out)
    print(study.json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
