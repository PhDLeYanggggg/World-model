"""Decompose the direct matched retrieval contrast on one full-grid denominator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_joint_retrieval as run

np, study, base = run.np, run.study, run.base
effects, frontier = run.report.effects, run.report.frontier


def components(after, before, target, valid, q):
    terms = []
    for idx in (np.flatnonzero(q == x) for x in np.unique(q)):
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        mask = valid[a] & valid[b]
        y = np.where(mask[..., None], target[a]-target[b], 0.)
        n, o = after[a]-after[b], before[a]-before[b]
        improvement = np.linalg.norm(o-y, axis=-1)-np.linalg.norm(n-y, axis=-1)
        known = np.where(mask, improvement, 0.).mean()
        debit = np.where(mask, 0., np.linalg.norm(n-o, axis=-1)).mean()
        terms.append([known, debit, known-debit])
    if not terms:
        return None
    result = np.mean(terms, axis=0)
    expected = frontier.aggregate_pair(after, before, target, valid, q, frontier.denominators(q, valid))[1]
    np.testing.assert_allclose(result[-1], expected, atol=1e-10)
    return result.tolist()


def main():
    path = run.OUT/"summary.json"
    parent = study.read(path)
    f = study.read(study.checked(parent["freeze"]))
    key, cohorts, _ = run.register()
    if f["registration"] != key or parent["registration"] != key or f["held_targets_used"]:
        raise ValueError("Require the sealed matched comparison")
    data = run.pair.run.turn.exp.loader(cohorts)
    sites = {}
    for s in run.old.SITES:
        d = data[s]
        rows = []
        for seed in run.old.SEEDS:
            z = run.old.rank.checked_npz(f["folds"][f"{s}_{seed}"]["held"]["forecast"])
            for rec in np.unique(d["rec"]):
                at = d["rec"] == rec
                if not (z["feasibility8"][at] != z["cv"][at]).any():
                    continue
                if not frontier.denominators(d["q"][at], d["v"][at]).all():
                    continue
                val = components(z["joint8"][at], z["feasibility8"][at], d["y"][at], d["v"][at], d["q"][at])
                if val is not None:
                    rows.append((str(rec), val))
        sites[s] = effects.site_means(rows)
    interval = effects.interval(list(sites.values()))
    np.testing.assert_allclose(interval["site_mean"][-1], parent["direct_pair_native"]["site_mean"][-1], atol=1e-10)
    out = dict(summary=study.artifact(path), result_source="fresh_run", decomposition=interval, by_location=sites,
        columns=["known_contribution_on_full_grid", "missing_step_debit", "full_grid_lower_gain"],
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"debit_audit.json", out)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
