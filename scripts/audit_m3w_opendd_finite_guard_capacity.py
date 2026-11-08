"""Exact minimum-risk menu at the same recipients and peak radii."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_finite_guard as study
from src.evaluation.m3w_opendd_location import paired

parent, run, np, torch = study.parent, study.run, study.np, study.torch
MENUS = {"original": ("half", "previous_shape", "tangent"),
         "expanded": ("half", "previous_shape", "tangent", "finite_guard", "curvature_guard")}


def best_risk(cv, options, y, valid):
    """Minimize observed ADE, break ties using the full-grid CV lower score."""
    bank = np.stack(options, axis=1)
    known = valid.any(1)
    clean = np.where(valid[..., None], y, 0.)
    distance = np.linalg.norm(bank-clean[:, None], axis=-1)
    ade = (distance*valid[:, None]).sum(2)/valid.sum(1).clip(1)[:, None]
    lower = np.stack([paired(cv, p, y, valid)[1].sum(1) for p in options], axis=1)
    # On wholly unknown rows risk is fixed by the common peak radius.
    loss = np.where(known[:, None], ade, 0.)
    best = np.where(loss == loss.min(1, keepdims=True), lower, -np.inf).argmax(1)
    return bank[np.arange(len(bank)), best]


def evaluate(d, values, radius, cuts, fraction):
    cv = d["cv"]
    result = {}
    for k, menu in MENUS.items():
        options = [cv+fraction*(values[n]-cv) for n in menu]
        oracle = best_risk(cv, options, d["y"], d["valid"])
        np.testing.assert_allclose(np.linalg.norm(oracle-cv, axis=-1).max(1), fraction*radius,
                                   atol=1e-14, rtol=1e-7)
        v = parent.policy(oracle, cv, cv, d["y"], d["valid"], d["recordings"],
                          easy_cut=cuts[0], hard_cut=cuts[1])
        compare = {a: parent.old.base.paired_summary(d, oracle, ref, radius > 0.)["lower_pp"]
                   for a, ref in (("cv", cv), ("tangent", options[2]))}
        rows = v["recordings"].values()
        result[k] = dict(pass_views=sum(r["supported"] for r in rows), views=len(rows),
            undefined=sum(r["completion_easy_risk_upper"] is None for r in rows),
            absent=sum(r["known_easy_net_degradation"] is None for r in rows), conservative=compare)
    if result["expanded"]["pass_views"] < result["original"]["pass_views"]:
        raise ValueError("Nested minimum-risk menu cannot lose a pass")
    return result


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, old_key, _, previous, warm = study.register()
    a = study.OUT / "analysis.json"
    origin = run.read(a)
    if origin["registration"] != key or origin["confirmation_access"]:
        raise ValueError("Require full registered matched fit")
    f = study.OUT / "capacity_registration.json"
    run.once(f, run.stable(dict(analysis=run.artifact(a), code=run.artifact(Path(__file__)), menus=MENUS,
        labels_for_diagnostic_only=True, original_2pct=True, exact_recipients_and_radius=True,
        confirmation_access=False)))
    outer = parent.old.frozen.load_networks(torch, parent.producer)
    sources = {s: parent.old.site_data(s, torch, parent.producer, parent.shape) for s in parent.old.SITES}
    fitted = {Path(r["path"]).stem: r for r in origin["fits"]}
    data = {}
    for pos, site in enumerate(parent.old.SITES):
        d = sources[site]
        cuts = parent.old.base.cuts_for([sources[s] for s in parent.old.SITES if s != site])
        data[site] = []
        for seed in parent.job.SEEDS:
            c = parent.old.candidate(d, outer[seed][pos], torch, parent.producer)
            p = warm[f"{site}_{seed}_tangent"]
            net, _ = study.load_warm(p, old_key, site, seed, c["x"].shape[1])
            models = {"tangent": (net, cuts, p)}
            for k in study.ARMS:
                p = fitted[f"{site}_{seed}_{k}"]
                state = torch.load(run.checked(p), map_location="cpu", weights_only=True)
                ident = state["identity"]
                if (ident["key"], ident["site"], ident["arm"], ident["seed"]) != (key, site, k, seed):
                    raise ValueError("Model roles changed")
                net = parent.model.CVTangent(c["x"].shape[1])
                net.load_state_dict(state["model"])
                models[k] = (net.eval(), cuts, p)
            models["finite"] = models["continued"]
            values, r = parent.paths(d, c, models, site, seed, previous)
            data[site].append(dict(seed=seed, half=evaluate(d, values, r, cuts, 1.),
                                   quarter=evaluate(d, values, r, cuts, .5)))
    summary = {}
    for part in ("half", "quarter"):
        summary[part] = {}
        for k in MENUS:
            by_site = [[r[part][k] for r in items] for items in data.values()]
            rows = [r for items in by_site for r in items]
            aggregate = {n: sum(r[n] for r in rows) for n in ("pass_views", "views", "undefined", "absent")}
            aggregate["conservative"] = {n: float(np.mean([np.mean([r["conservative"][n] for r in b])
                for b in by_site])) for n in ("cv", "tangent")}
            summary[part][k] = aggregate
    run.once(study.OUT / "capacity.json", run.stable(dict(registration=run.sha(f), summary=summary,
        by_location=data, optimum_for_risk_not_utility=True, no_refit=True,
        confirmation_access=False, deployment=False)))


if __name__ == "__main__":
    main()
