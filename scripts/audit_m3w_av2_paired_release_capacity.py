"""Count AV2 opportunities left by the fixed comparator and original release rule."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_supported_gain as run

prev, np, study, base = run.prev, run.np, run.study, run.base
KEYS = ("two_slots", "strict_joint_available", "fixed_additive_eligible", "paired_release_available",
        "fixed_additive_risk_pass", "fixed_additive_easy_undefined", "fixed_additive_dual_gain",
        "complete", "complete_strict_joint_available", "complete_paired_release_available",
        "any_dual_risk_action", "any_dual_risk_pair", "complete_any_dual_risk_pair")


def eligible_additive(m):
    return bool(m["risk"]["supported"] and m["gain"] is not None and min(m["gain"]) > 0.)


def alternative_capacity(d, paths, y, v, cut):
    """A ceiling over any two lawful slates, not a learned additive policy."""
    c, o = d["cv"], d["ols4"]
    e = np.where(v, np.linalg.norm(c-y, axis=-1), 0.)
    mass = e.sum()
    if mass <= 0.:
        return False, False
    e = e.sum(1)/v.sum(1).clip(1)
    cap = .02*e[(e > 0.) & (e <= cut[0])].sum()
    t = prev.pair.audit.terms(paths, c, o, c, y, v, cut[0], 1., mass, 1)
    safe = paths[(t[:, -1] > 0.) & (t[:, 1:3] > 0.).all(1) & (t[:, 4:6] <= 0.).all(1) & (t[:, 6] <= cap)]
    if len(safe) < 2:
        return bool(len(safe)), False
    # Evaluate both the relative path and the individual trajectory contrast
    # directly, since separately debited bounds do not subtract.
    for a in safe:
        rel = run.av.frontier.pair_reduction(safe, a, y, v)[:, 1] > 1e-10
        if not rel.any():
            continue
        ea = np.linalg.norm(a-y, axis=-1)
        ej = np.linalg.norm(safe[rel]-y, axis=-1)
        debit = np.linalg.norm(safe[rel]-a, axis=-1)
        trajectory = np.where(v, ea-ej, -debit).sum((1, 2))*100./mass
        if (trajectory >= -1e-8).any():
            return True, True
    return True, False


def audit(d, seed, truth, cut):
    y, v = truth
    a, _, paths, feat = prev.candidates(d, seed)
    if feat is None:
        return {}
    _, strict, _ = prev.label_terms(d, paths, a, y, v, cut)
    m = prev.metrics(a, d, y, v, cut)
    add = eligible_additive(m)
    both = bool(add and strict.any())
    any_action, any_pair = alternative_capacity(d, paths, y, v, cut)
    if both and not any_pair:
        raise ValueError("A fixed eligible pair must also be in the relaxed ceiling")
    return dict(two_slots=1, strict_joint_available=int(strict.any()), fixed_additive_eligible=int(add),
        fixed_additive_risk_pass=int(m["risk"]["supported"]),
        fixed_additive_easy_undefined=int(m["risk"]["completion_easy_risk_upper"] is None),
        fixed_additive_dual_gain=int(m["gain"] is not None and min(m["gain"]) > 0.),
        paired_release_available=int(both), complete=int(v.all()),
        complete_strict_joint_available=int(v.all() and strict.any()),
        complete_paired_release_available=int(v.all() and both), any_dual_risk_action=int(any_action),
        any_dual_risk_pair=int(any_pair), complete_any_dual_risk_pair=int(v.all() and any_pair))


def main():
    key, original, source, seal, _, _ = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    freeze = study.read(study.checked(result["freeze"]))
    if result["registration"] != key or freeze["registration"] != key or freeze["held_labels_used"]:
        raise ValueError("Require the sealed supported-gain study")
    data = prev.read_inputs(original, seal)
    truth = prev.load_targets(source, data, data)
    folds, totals = {}, {r: Counter() for r in ("calibration", "held")}
    for name, z in freeze["folds"].items():
        ident = z["identity"]
        folds[name] = {}
        for role, total in totals.items():
            city = ident[role]
            c = Counter()
            for n, d in data.items():
                if d["city"] == city:
                    c.update(audit(d, ident["seed"], truth[n], z["cut"]))
            total.update(c)
            folds[name][role] = {k: c[k] for k in KEYS}
    if totals["held"]["strict_joint_available"] != result["capacity"]["oracle"]:
        raise ValueError("Strict two-slot oracle count changed")
    out = dict(source=study.artifact(p), code=study.artifact(Path(__file__)), result_source="fresh_run",
        by_fold=folds, totals={r: {k: c[k] for k in KEYS} for r, c in totals.items()},
        relaxed_reference="Any two qualifying exact-slot slates; trajectory noninferiority and full-grid pair lift evaluated directly.",
        exact_hindsight_only=True, no_policy_or_protocol_change=True, independent_confirmation=False,
        reserved_outcomes_opened=False)
    base.immutable(run.OUT/"paired_release_capacity.json", out)
    print(study.json.dumps(out["totals"], indent=2))


if __name__ == "__main__":
    main()
