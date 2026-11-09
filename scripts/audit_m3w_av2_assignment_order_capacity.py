"""Read-only exact opportunity audit against the unchanged learned additive path."""

from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_assignment_order as run

np, study = run.np, run.study
UTILITY_TOL_PP = 1e-8
PAIR_TOL = 1e-10


def contrasts(paths, soft, cv, y, v):
    norm = lambda z: np.linalg.norm(z, axis=-1)
    truth = np.where(v[..., None], y, 0.)
    mass = np.where(v, norm(cv-truth), 0.).sum()
    if mass <= 0.:
        raise ValueError("A qualifying slate must have positive reference error")
    known = np.where(v, norm(soft-truth)-norm(paths-truth), 0.).sum((1, 2))*100./mass
    debit = np.where(v, 0., norm(paths-soft)).sum((1, 2))*100./mass
    pair = run.av.frontier.pair_reduction(paths, soft, y, v)
    return known, known-debit, pair


def main():
    key, data, src, _, _, _, control, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    summary = study.read(path)
    freeze = study.read(study.checked(summary["freeze"]))
    if (freeze["registration"] != key or freeze["held_labels_used"] or freeze["reserved_outcomes_opened"]
            or summary["independent_confirmation"] or summary["reserved_outcomes_opened"]):
        raise ValueError("Require scored development experiment")
    truth = run.prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    cities = defaultdict(Counter)
    for name, ref in freeze["folds"].items():
        f = study.read(study.checked(ref))
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        parent = study.read(study.checked(f["identity"]["parent"]))
        counts = cities[held]
        for (n, d, p, opts, q), row in zip(run.prior.views(data, control["folds"][name], held, seed, parent),
                                          f["rows"], strict=True):
            if row["clip"] != n or row["hash"] != q["hash"]:
                raise ValueError("Wrong frozen choices")
            y, v = truth[n]
            cv, ols = d["cv"], d["ols4"]
            ok, _ = run.bank.model.eligibility(cv, ols, p, y, v, cut[0], opts)
            counts["views"] += 1
            if not ok.any():
                continue
            counts["standalone_opportunity"] += 1
            ix = np.flatnonzero(ok)
            soft_ix = row["choices"][0][0]
            paths = np.stack([run.bank.model.render(cv, p, opts[k]) for k in ix])
            soft = run.bank.model.render(cv, p, opts[soft_ix])
            known, gain, pair = contrasts(paths, soft, cv, y, v)
            known_both = (known >= -UTILITY_TOL_PP) & (pair[:, 0] > PAIR_TOL)
            both = (gain >= -UTILITY_TOL_PP) & (pair[:, 1] > PAIR_TOL)
            if (both & ~known_both).any():
                raise ValueError("A conservative witness must also be a known-step witness")
            counts["soft_eligible"] += int(ok[soft_ix])
            counts["qualified_better_trajectory"] += int((gain > UTILITY_TOL_PP).any())
            counts["qualified_known_joint"] += int(known_both.any())
            counts["qualified_conservative_joint"] += int(both.any())
            counts["both_arms_eligible_conservative_joint"] += int(ok[soft_ix] and both.any())
            counts["both_arms_eligible_conservative_joint_complete"] += int(ok[soft_ix] and both.any() and v.all())
            for arm, (i, _) in zip(run.ARMS[1:], row["choices"][1:], strict=True):
                counts[f"{arm}_realized_joint"] += int((ix[both] == i).any())
                counts[f"{arm}_realized_joint_both_eligible"] += int(ok[soft_ix] and (ix[both] == i).any())
    total = sum(cities.values(), Counter())
    if (total["views"] != summary["views"] or total["standalone_opportunity"] != summary["opportunities"]
            or total["soft_eligible"] != summary["arms"]["soft"]["eligible"]):
        raise ValueError("Exact bank or baseline failed to replay")
    run.base.immutable(run.OUT/"matched_capacity.json", dict(source=study.artifact(path), result_source="fresh_run",
        outcome_only=True, used_for_selection=False, independent_confirmation=False,
        reserved_outcomes_opened=False, counts=total, by_city=cities))
    print(study.json.dumps(total, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
