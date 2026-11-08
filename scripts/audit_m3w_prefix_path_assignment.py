"""Compare prefix assignment to exact same-count random allocation utility."""

from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_prefix_path_replay as study

run, np, torch, prior, source = study.run, study.np, study.torch, study.prior, study.source


def shuffled(cv, bank, choice, y, valid):
    """Expected additive utility over permutations of the same action multiset."""
    counts = np.bincount(choice, minlength=bank.shape[1])
    prob = counts/counts.sum()
    chosen = bank[np.arange(len(choice)), choice]
    cv_pair = [prior.paired(cv, path, y, valid)[1].sum() for path in bank.swapaxes(0, 1)]
    direct = [prior.paired(path, chosen, y, valid) for path in bank.swapaxes(0, 1)]
    return np.array([prior.paired(cv, chosen, y, valid)[1].sum(), prob @ cv_pair,
                     prob @ [a.sum() for a, _ in direct], prob @ [b.sum() for _, b in direct]])


def cell(cv, bank, on, choice, y, valid, rec):
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    names = np.unique(rec[on])
    v = np.zeros(4)
    for name in names:
        m = on & (rec == name)
        v += shuffled(cv[m], bank[m], choice[m], y[m], valid[m])
    return dict(zip(("policy_vs_cv", "shuffle_vs_cv", "direct_observed", "direct_lower"), (100*v/mass).tolist(), strict=True))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _ = study.register()
    p = study.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or old["confirmation_access"]:
        raise ValueError("Require the completed exposed replay")
    registration = study.OUT / "assignment_registration.json"
    run.once(registration, run.stable(dict(parent=run.artifact(p), code=run.artifact(Path(__file__)),
        null="all permutations within each recording and seed at fixed radii and six action counts",
        exact_expected_utility=True, no_risk_claim_for_random_control=True, confirmation_access=False)))
    nets = prior.frozen_models(*frozen[:5])
    out, summary = {}, {}
    for domain, groups in old["groups"].items():
        out[domain] = {}
        names = source.SITES if domain == "opendd" else source.ROSTER
        for j, name in enumerate(names):
            if domain == "opendd":
                with np.load(source.prior.PRIVATE / f"{name}_cohort.npz", allow_pickle=False) as z:
                    h, y, rec = z["history"], z["future"], np.repeat(z["recording"], np.diff(z["offsets"]))
            else:
                filename, _ = source.ROSTER[name]
                h, y, _ = source.windows(source.read_tracks(source.SOURCE / filename)[0])
                rec = np.full(len(y), name)
            valid = np.isfinite(y).all(2)
            rows = []
            for i, seed in enumerate(study.SEEDS):
                member = [nets[seed][j]] if domain == "opendd" else nets[seed]
                pred, _, choice = study.infer(h, partial(prior.forecast, models=member))
                paths, bank, on = pred["half"]
                z = {k: cell(paths["cv"], bank, on, choice[k], y, valid, rec) for k in study.METHODS}
                for k in z:
                    np.testing.assert_allclose(z[k]["policy_vs_cv"], groups[name][i]["half"]["arms"][k]["lower"]["cv"], atol=1e-12)
                rows.append(z)
            out[domain][name] = rows
        rows = out[domain].values()
        n = len(rows)
        draws = np.random.default_rng(817).integers(n, size=(3000, n)) if domain == "opendd" else None
        summary[domain] = {}
        for k in study.METHODS:
            summary[domain][k] = {}
            for m in next(iter(rows))[0][k]:
                a = np.array([np.mean([r[k][m] for r in v]) for v in rows])
                summary[domain][k][m] = dict(point=float(a.mean()), interval95=np.quantile(a[draws].mean(1), [.025, .975]).tolist()
                                            if draws is not None else None)
    run.once(study.OUT / "assignment.json", run.stable(dict(registration=run.sha(registration), summary=summary, groups=out,
        no_training=True, risk_not_estimated_for_shuffle=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
