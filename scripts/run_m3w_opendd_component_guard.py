"""Attribute comoving forecast gains and test a matched component guard."""

from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_comoving_eqmotion as prior
from src.world_model import m3w_opendd_component_guard as model

run, parent, paired, ledger, two = prior.run, prior.parent, prior.paired, prior.ledger, prior.two
OUT = prior.OUT.parent / "opendd_component_guard_v1"
FIELDS = ("edits", "views", "undefined", "risk_pass", "dual_positive")


def register():
    key, a, ph, h, old, source = prior.register()
    p = prior.OUT / "analysis.json"
    previous = run.read(p)
    if previous["registration"] != key or not previous["original_2pct"] or previous["confirmation_access"]:
        raise ValueError("Require frozen comoving forecasts on exposed OpenDD only")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=run.artifact(p), native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        primary="direct missing-step lower utility: guard versus full direction at the same row peak",
        guard="retain whole longitudinal or lateral residual only if its terminal sign agrees with OLS4 minus CV",
        axis="last observed displacement, else first-to-last chord, else no guard",
        strata="original donor easy/hard cuts; separate donor quartiles of observed normalized CV-to-OLS4 disagreement",
        matched="original query/recipient/radius; when a shape is zero, emit the original shape",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        no_training=True, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, old, previous, source


def raw(d, bank, paths, cuts, drift, drift_cuts):
    cv, ols = bank[:2]
    y, valid = d["y"], d["valid"]
    floor = two.moments(cv, cv, y, valid, cuts[0])
    base, known = floor["base"], floor["known"]
    easy = floor["easy"]
    hard = known & (base > cuts[1])
    groups = dict(easy=easy, hard=hard, other_observed=known & ~(easy | hard), no_label=~known,
                  stable=drift <= drift_cuts[0], mixed=(drift > drift_cuts[0]) & (drift <= drift_cuts[1]),
                  variable=drift > drift_cuts[1])
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    names = ("full", "longitudinal", "lateral", "guard")
    labels = [paired(cv, paths[k], y, valid) for k in names]
    win = np.stack([np.zeros(len(cv)), *[v[1].sum(1) for v in labels]], axis=1).argmax(1)
    paths["oracle"] = np.stack([cv, *[paths[k] for k in names]], axis=1)[np.arange(len(cv)), win]
    out = {}
    full, endpoint = valid.all(1), valid[:, -1]
    for k, p in paths.items():
        observed, lower = paired(cv, p, y, valid)
        err = np.linalg.norm(p-y, axis=-1)
        m = two.moments(cv, p, y, valid, cuts[0])
        easy_mass = m["ref_easy"].sum()
        out[k] = dict(known_vs_cv=float(100.*observed.sum()/mass), lower_vs_cv=float(100.*lower.sum()/mass),
            lower_vs_ols4=float(100.*paired(ols, p, y, valid)[1].sum()/mass),
            missing_debit=float(100.*(observed-lower).sum()/mass),
            easy_net_percent=float(100.*m["easy_signed"].sum()/easy_mass) if easy_mass > 0. else None,
            complete_ade_sum=float(err[full].sum()/12.), endpoint_fde_sum=float(err[endpoint, -1].sum()),
            strata={s: dict(known_pp=float(100.*observed[ix].sum()/mass),
                            lower_pp=float(100.*lower[ix].sum()/mass)) for s, ix in groups.items()})
    effects = {"guard_vs_equal_peak": float(100.*paired(paths["equal_peak"], paths["guard"], y, valid)[1].sum()/mass)}
    return dict(policies=out, effects=effects, sizes={k: int(v.sum()) for k, v in groups.items()},
                complete=int(full.sum()), endpoint=int(endpoint.sum()))


def matched(d, bank, packet, menu, count, offsets, cuts, archived):
    cv, ols = bank[:2]
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    anchor, prefix = packet["paths"][at, arm, 0], packet["paths"][at, arm, 3]
    r = model.peak(anchor-cv[at])
    offered = dict(anchor=anchor, prefix_match=prefix)
    fallbacks = {}
    for k in ("full", "longitudinal", "lateral", "guard"):
        v = offsets[k][at]
        offered[k] = prior.radius.match_radius(cv[at], anchor, v[:, None])[:, 0]
        fallbacks[k] = int((model.peak(v) <= 1e-12).sum())
    forecasts, metrics = {}, {}
    for k, p in offered.items():
        f = cv.copy()
        f[at] = p
        np.testing.assert_array_equal(np.flatnonzero((f != cv).any((1, 2))), np.sort(at))
        np.testing.assert_allclose(model.peak(p-cv[at]), r, atol=1e-10, rtol=1e-8)
        metrics[k] = parent.score(d, cv, ols, f, cuts)
        forecasts[k] = f
    for k, ref in (("anchor", "anchor"), ("prefix_match", "prefix_match"), ("full", "comoving_ego")):
        for field, v in archived["policies"][ref].items():
            np.testing.assert_allclose(metrics[k][field], v, rtol=0., atol=1e-10)
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {f"{k}_vs_{ref}": float(100.*paired(forecasts[ref], forecasts[k], y, valid)[1].sum()/mass)
               for k in ("longitudinal", "lateral", "guard") for ref in ("full", "prefix_match")}
    return dict(policies=metrics, effects=effects, fallbacks=fallbacks)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, a, ph, h, _, previous, source = register()
    sites = {s: prior.load(s, source)[:3] for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(v[0]) for s, v in sites.items()}
    drift = {s: model.instability(v[0]["h"], *banks[s][:2]) for s, v in sites.items()}
    result = {}
    for held, (d, rec, starts) in sites.items():
        donors = tuple(s for s in sites if s != held)
        cuts = np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in donors]), [.25, .75])
        dc = np.quantile(np.concatenate([drift[s] for s in donors]), [.25, .75])
        b = banks[held]
        packet = prior.prior.prepare({s: v[0] for s, v in sites.items()}, held, donors, b)
        result[held] = []
        for row in a["by_site"][held]:
            seed = row["seed"]
            ref = next(r for r in previous["by_site"][held] if r["seed"] == seed)
            with np.load(run.checked(ref["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            offsets, axis, along, across = model.split(d["h"], *b[:2], ego)
            p = {"cv": b[0], "ols4": b[1], **{k: b[0]+v for k, v in offsets.items()}}
            p["full"] = ego
            r = raw(d, b, p, cuts, drift[held], dc)
            np.testing.assert_allclose(r["policies"]["full"]["lower_vs_cv"],
                ref["raw"]["arms"]["comoving_ego"]["lower"]["cv"], atol=1e-12, rtol=0.)
            m, n = ledger.frozen_count(d, b, rec, starts, ph["heads"][f"{held}/{seed}/additive_rank"],
                                      h["heads"][f"{held}_{seed}_motion"])
            score = matched(d, b, packet, m, n, offsets, cuts, ref["matched"])
            result[held].append(dict(seed=seed, raw=r, matched=score, defined=int(axis.sum()),
                                     along=int(along.sum()), across=int(across.sum())))
        print(held, len(d["h"]), flush=True)
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    first = flat[0]
    names = first["raw"]["policies"]
    metrics = {}
    for k in names:
        rows = [r["raw"]["policies"][k] for r in flat]
        easy = [np.mean([r["raw"]["policies"][k]["easy_net_percent"] for r in v]) for v in result.values()
                if all(r["raw"]["policies"][k]["easy_net_percent"] is not None for r in v)]
        metrics[k] = dict(**{f: ci(lambda r: r["raw"]["policies"][k][f]) for f in (
            "known_vs_cv", "lower_vs_cv", "lower_vs_ols4", "missing_debit")},
            complete_ade=sum(r["complete_ade_sum"] for r in rows)/sum(r["raw"]["complete"] for r in flat),
            endpoint_fde=sum(r["endpoint_fde_sum"] for r in rows)/sum(r["raw"]["endpoint"] for r in flat),
            easy_net_site_range=[float(min(easy)), float(max(easy))],
            strata={s: ci(lambda r: r["raw"]["policies"][k]["strata"][s]["lower_pp"])
                    for s in first["raw"]["sizes"]})
    matched_totals = {k: {f: sum(r["matched"]["policies"][k][f] for r in flat) for f in FIELDS}
                      for k in first["matched"]["policies"]}
    matched_effects = {k: ci(lambda r: r["matched"]["effects"][k]) for k in first["matched"]["effects"]}
    summary = dict(raw=metrics, matched=matched_totals, matched_effects=matched_effects,
        raw_effects={k: ci(lambda r: r["raw"]["effects"][k]) for k in first["raw"]["effects"]},
        unique_slice_rows={k: sum(v[0]["raw"]["sizes"][k] for v in result.values()) for k in first["raw"]["sizes"]},
        matched_fallbacks={k: sum(r["matched"]["fallbacks"][k] for r in flat) for k in first["matched"]["fallbacks"]})
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, by_site=result, summary=summary,
        no_training=True, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
