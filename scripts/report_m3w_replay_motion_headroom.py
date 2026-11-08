"""Post-hoc capacity of the finite frozen expert menu on fitting sources."""

from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_replay_motion as fit

np = fit.np


def diagnostic(d, queries, delta, floor, cv, weights):
    y, valid = d["target"], d["valid"]
    clean = np.where(valid[..., None], y, 0.)
    floor_error = np.linalg.norm(floor - clean, axis=-1)
    cv_error = np.linalg.norm(cv - clean, axis=-1)
    count = valid.sum(1)
    w = np.zeros(len(y))
    for q in queries:
        w[q] = 1. / len(q)
    sites, src = np.unique(d["sites"], return_inverse=True)

    def mass(value):
        return np.bincount(src, weights=w * value, minlength=len(sites))

    denom = mass(np.where(valid, cv_error, 0.).sum(1) / count.clip(1))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))

    def components(change):
        err = np.linalg.norm(floor + change - clean, axis=-1)
        credit = np.where(valid, floor_error - err, 0.).mean(1)
        debit = np.where(valid, 0., np.linalg.norm(change, axis=-1)).mean(1)
        return credit, debit

    report = {a: [] for a in ("known_full_grid_credit", "missing_step_debit",
        "all_step_lower", "complete12_contribution", "incomplete_contribution")}
    for arm in weights[:, 0]:
        credit, debit = components(np.einsum("nk,nkhd->nhd", arm, delta))
        lower = credit - debit
        for key, values in zip(report, (credit, debit, lower, lower * (count == 12),
                lower * (count < 12)), strict=True):
            report[key].append(100. * mass(values) / denom)
    intervals = fit.parent.old.metric.intervals
    learned = {key: intervals(np.stack(values, axis=1), draws) for key, values in report.items()}

    names = list(fit.model.EXPERTS)
    pairs = list(combinations(range(len(names)), 2))
    bank = [delta[:, i] for i in range(len(names))]
    bank.extend((delta[:, a] + delta[:, b]) * .5 for a, b in pairs)
    names.extend(f"{fit.model.EXPERTS[a]}+{fit.model.EXPERTS[b]}" for a, b in pairs)
    value = np.stack([c - m for c, m in map(components, bank)], axis=1)
    selected = value.argmax(axis=1)
    oracle = value[np.arange(len(y)), selected]
    if (oracle < -1e-10).any():
        raise ValueError("OLS4 identity must be in the oracle menu")
    return dict(replay_decomposition=learned,
        finite_menu_hindsight_lower=intervals((100. * mass(oracle) / denom)[:, None], draws),
        oracle_positive_rows=int((oracle > 1e-8).sum()),
        oracle_choices=dict(zip(names, np.bincount(selected, minlength=len(names)).tolist(), strict=True)),
        source_groups=len(sites), rows=len(y), complete12_rows=int((count == 12).sum()),
        interpretation="fitting-source diagnostic; not learned, not an upper bound on all mixtures",
        calibration_or_confirmation_used=False)


def main():
    _, jobs = fit.register()
    frozen = fit.prev.receipt(fit.PUBLIC, "train")
    main_score = fit.prev.receipt(fit.PUBLIC, "replay")
    d = fit.parent.data(jobs, lambda **_: None)
    queries = fit.parent.old.fitting_layout(jobs[0]["identity"], d)
    _, _, delta, _, paths = fit.model.design(d["geometry"][:, :16].reshape(-1, 8, 2))
    with np.load(fit.run.checked(frozen["weights"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        weights = z["weights"].astype(np.float64)
    result = diagnostic(d, queries, delta, paths["ols4"], paths["cv"], weights)
    np.testing.assert_allclose(result["replay_decomposition"]["all_step_lower"]["point"],
        main_score["contrasts"]["replay_vs_ols4"]["full_grid_lower"]["point"], atol=1e-10)
    result["frozen_score_sha256"] = fit.run.sha(fit.PUBLIC / "analysis.json")
    result["implementation_sha256"] = fit.run.sha(Path(__file__))
    fit.run.once(fit.PUBLIC / "headroom.json", result)


if __name__ == "__main__":
    main()
