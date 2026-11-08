"""Separate adding an OLS direction from allowing a larger displacement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_optional_reference_bridge as study

run, np, torch, transfer = study.run, study.np, study.torch, study.transfer


def clipped_bank(cv, ols, half):
    step = ols-cv
    radius = np.linalg.norm(step, axis=-1).max(1)
    alpha = np.divide(half, radius, out=np.zeros_like(half), where=radius > 0.)
    scale = np.minimum(alpha[:, None], study.MIX)
    bridges = cv[:, None]+scale[..., None, None]*step[:, None]
    # Keep the original endpoint exactly when the radius does not clip it.
    bridges[scale[:, -1] == 1., -1] = ols[scale[:, -1] == 1.]
    if np.any(np.linalg.norm(bridges-cv[:, None], axis=-1).max(2) > half[:, None]+1e-12):
        raise ValueError("Clipped bridge exceeded its own row radius")
    return bridges


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, intake, scale = study.register()
    p = study.OUT / "analysis.json"
    previous = run.read(p)
    if previous["registration"] != key or previous["confirmation_access"]:
        raise ValueError("Require the completed exposed capacity analysis")
    r = study.OUT / "clipped_registration.json"
    run.once(r, run.stable(dict(parent=run.artifact(p), code=run.artifact(Path(__file__)),
        each_new_arm_at_most_original_half_radius=True, same_cv_risk=True,
        labels_hindsight_only=True, confirmation_access=False)))
    nets = transfer.frozen_models(*frozen[:5])
    sources = study.source_data(intake, scale)
    easy = {}
    for k, (h, y, _) in sources["opendd"].items():
        cv = transfer.motion.rollout(h, 12)[0]
        ok = np.isfinite(y).all((1, 2))
        mass = np.linalg.norm(cv[ok]-y[ok], axis=-1).mean(1)
        easy[k] = mass[mass > 0.]
    rows = {}
    for domain, groups in sources.items():
        rows[domain] = {}
        for j, (site, (h, y, rec)) in enumerate(groups.items()):
            rows[domain][site] = []
            valid = np.isfinite(y).all(2)
            cut = np.quantile(np.concatenate([v for k, v in easy.items() if k != site]), [.25, .75])
            for i, seed in enumerate(study.previous.choice_study.SEEDS):
                members = [nets[seed][j]] if domain == "opendd" else nets[seed]
                cv, ols, base, _, delta = transfer.forecast(h, members)
                paths, _ = transfer.matched_paths(cv, base, delta, .5)
                half = .5*np.linalg.norm(base-cv, axis=-1).max(1)
                bank = np.stack([paths[k] for k in transfer.ARMS], axis=1)
                arms = np.concatenate((bank, clipped_bank(cv, ols, half)), axis=1)
                old = previous["groups"][domain][site][i]
                if old["seed"] != seed:
                    raise ValueError("Seed alignment changed")
                local = {}
                for rec_id, b in old["recordings"].items():
                    at = rec == rec_id
                    a = study.assess(cv[at], ols[at], arms[at], y[at], valid[at], rec_id, cut)
                    if (b["frozen_optional"]["classification"] == "verified_positive_both"
                            and a["classification"] not in ("verified_positive_both", "unresolved")):
                        raise ValueError("Adding the clipped bridge removed an existing witness")
                    local[rec_id] = dict(clipped=a, unrestricted_class=b["union"]["classification"],
                                         original_class=b["frozen_optional"]["classification"])
                rows[domain][site].append(dict(seed=seed, recordings=local))
    summary = {}
    for k, groups in rows.items():
        a = [row for cells in groups.values() for c in cells for row in c["recordings"].values()]
        key = "verified_positive_both"
        summary[k] = dict(views=len(a), positive_original=sum(v["original_class"] == key for v in a),
            positive_clipped=sum(v["clipped"]["classification"] == key for v in a),
            positive_unrestricted=sum(v["unrestricted_class"] == key for v in a),
            unresolved=sum(v["clipped"]["classification"] == "unresolved" for v in a),
            absent_easy=sum(v["clipped"]["classification"] == "absent_easy" for v in a))
    run.once(study.OUT / "clipped.json", run.stable(dict(registration=run.sha(r), summary=summary, by_source=rows,
        oracle_only=True, no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
