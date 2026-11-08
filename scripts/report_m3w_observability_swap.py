"""Post-hoc support-head substitution with a frozen conditional-gain head."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_observability as fit

np, torch = fit.np, fit.torch


def main():
    _, jobs = fit.register()
    frozen = fit.prev.receipt(fit.PUBLIC, "train")
    official = fit.prev.receipt(fit.PUBLIC, "replay")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    d, qs, packed = fit.prior.data(jobs, lambda **_: None)
    x, _, scale, paths, delta, eligible, _, _ = packed
    quality, sources = fit.quality_inputs(jobs[0]["identity"], d, qs, lambda **_: None)
    if (fit.base.fit_source.source.array_hash(quality) != frozen["quality_hash"]
            or sources != frozen["quality_sources"]):
        raise ValueError("Feature bank differs from original fit")
    x = torch.from_numpy(np.column_stack((x, quality)))
    dist = np.linalg.norm(delta, axis=-1)
    debit = torch.from_numpy((dist/scale[:, None]).astype(np.float32))
    y, valid, floor = d["target"], d["valid"], paths["ols4"]
    sites, site = np.unique(d["sites"], return_inverse=True)
    weight = np.empty(len(y))
    for q in qs:
        weight[q] = 1. / len(q)

    def mass(z):
        return np.bincount(site, weights=weight*z, minlength=len(sites))

    cv = fit.prior.prior.ade(paths["cv"], y, valid)[0]
    denom = mass(cv)
    with np.load(fit.run.checked(frozen["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        original, support, prevalence = (z[k] for k in ("scores", "support", "prevalence"))
    i = fit.model.ARMS.index("factor_quality")
    motion_i = fit.model.ARMS.index("factor_motion")
    forecasts, direct, effects = {}, {}, {}
    for k, seed in enumerate(fit.prev.SEEDS):
        mu = np.empty_like(support[k, i])
        visited = 0
        for row in frozen["fits"]:
            ident = row["identity"]
            if ident["arm"] != "factor_quality" or ident["seed"] != seed:
                continue
            ix = np.flatnonzero(d["sites"] == ident["held"])
            state = torch.load(fit.run.checked(row["checkpoint"]), map_location="cpu", weights_only=False)
            if state["identity"] != ident or state["step"] != fit.STEPS:
                raise ValueError("Frozen checkpoint changed")
            net = fit.model.network(x.shape[-1]).eval()
            net.load_state_dict(state["model"])
            with torch.inference_mode():
                out = net(x[ix])
                u, p = fit.model.estimate(out, debit[ix], ident["arm"])
                mu[ix] = out[..., 13:].tanh().numpy()
            np.testing.assert_array_equal(p.numpy(), support[k, i, ix])
            np.testing.assert_array_equal(u.numpy()*scale[ix], original[k, i, ix])
            visited += len(ix)
        if visited != len(y):
            raise ValueError("No conditional prediction for some rows")
        rank = {"quality_support": original[k, i]}
        for name, p in (("motion_support", support[k, motion_i]), ("prevalence", prevalence),
                        ("future_mask_hindsight", valid)):
            rank[name] = ((p*mu-(1.-p))*dist).mean(1)
        chosen = {a: fit.prior.model.choose(s, eligible, qs) for a, s in rank.items()}
        reference = chosen["quality_support"]
        paths_out = {a: floor + delta*s[:, None, None] for a, s in chosen.items()}
        count = [int(reference[q].sum()) for q in qs]
        for name, take in chosen.items():
            np.testing.assert_array_equal([take[q].sum() for q in qs], count)
            p = paths_out[name]
            lower = fit.prior.prior.prior.scores(p, floor, y, valid)["full_grid_lower"]
            forecasts.setdefault(name, []).append(100.*mass(lower)/denom)
            contrast = fit.prior.prior.prior.scores(p, paths_out["quality_support"], y, valid)["full_grid_lower"]
            direct.setdefault(name, []).append(100.*mass(contrast)/denom)
            effects.setdefault(name, []).append(dict(changed_queries=sum(not np.array_equal(
                take[q], reference[q]) for q in qs), selected_unknown_step_fraction=float(
                (~valid[take]).mean())))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda x: fit.prior.prior.prior.parent.old.metric.intervals(np.stack(x, 1), draws)
    total = {a: interval(x) for a, x in forecasts.items()}
    np.testing.assert_allclose(total["quality_support"]["point"],
        official["utility"]["factor_quality"]["full_grid_lower"]["point"], atol=1e-10)
    result = dict(vs_ols4=total, direct_vs_quality_support={a: interval(x) for a, x in direct.items()},
        effects=effects, future_mask_case_hindsight_only=True, official_inputs_unchanged=True,
        conditional_forecast_head_frozen=True, independent_confirmation=False,
        main_sha256=fit.run.sha(fit.PUBLIC / "analysis.json"), implementation_sha256=fit.run.sha(Path(__file__)))
    fit.run.once(fit.PUBLIC / "support_swap_diagnostic.json", result)


if __name__ == "__main__":
    main()
