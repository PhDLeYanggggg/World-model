"""Rescale the existing CV-tangent forecast to the same selected rows and radii."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_shape_menu_value as study

parent, base, run, torch = study.parent, study.base, study.run, study.torch
peak = study.model.menu.peak


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, h, _ = study.register()
    p, ck = (study.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    old, heads = run.read(p), run.read(ck)
    if (old["registration"] != key or heads["registration"] != key or old["heads"] != run.sha(ck)
            or old["confirmation_access"] or not old["exact_rows_and_radii"]):
        raise ValueError("Require the frozen same-radius models")
    _, ref = parent.register()
    nets = parent.transfer.frozen_models(*ref[:5])
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    result = {}
    for j, (held, d) in enumerate(sites.items()):
        cuts = np.quantile(np.concatenate([v["cut_samples"] for s, v in sites.items() if s != held]), [.25, .75])
        bank = parent.prior.analytic(d)
        cv, ols, b = bank[:3]
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        y, valid = d["y"], d["valid"]
        mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            st = study.study
            rk = st.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: st.prior.model.SwapValue(False), st.prior.STEPS)
            m = st.menus(d, bank, rk)
            x, _, _ = base.features(d, bank, m)
            net = st.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                score = base.model.score(net(torch.from_numpy(x["motion"])).numpy(), risk=False)
            count = st.model.streaming_count(score, m["legal"], rec, starts)
            on = np.arange(2) < count[:, None]
            row, arm = m["row"][on], m["arm"][on]
            paths = {k: cv.copy() for k in ("anchor", "tangent", "prefix", "learned_dynamics")}
            nzero = 0
            if len(row):
                amp = np.array([.25, .5, 1.])[arm]
                x0, _, _, shape, geom, _, _ = study.model.features(d["h"][row], amp)
                head = st.load(heads["heads"][f"{held}_{seed}_dynamics"], lambda: study.model.Head(x0.shape[1], "dynamics"), study.STEPS)
                with torch.inference_mode():
                    proj = head(torch.from_numpy(x0)).numpy()
                choice = np.linalg.norm(geom-proj[:, None], axis=-1).mean(-1).argmin(1)
                prefix = study.model.menu.prefix_rank(d["h"][row], amp, np.ones(len(row), dtype=bool)).mean(0).argmax(1)
                paths["anchor"][row] = b[row, arm]
                paths["learned_dynamics"][row] = shape[np.arange(len(row)), choice]
                paths["prefix"][row] = shape[np.arange(len(row)), prefix]
                orig = parent.prior.prior.prepare(d, parent.transfer.forecast(d["h"], [nets[seed][j]]))["tangent"]
                z = orig[row]-cv[row]
                r = peak(z)
                ref_r = peak(paths["anchor"][row]-cv[row])
                gain = np.divide(ref_r, r, out=np.zeros_like(r), where=r > 1e-12)
                paths["tangent"][row] = np.where((r > 1e-12)[:, None, None], cv[row]+z*gain[:, None, None], paths["anchor"][row])
                nzero = int((r <= 1e-12).sum())
            active = (paths["anchor"] != cv).any((1, 2))
            for v in paths.values():
                np.testing.assert_array_equal((v != cv).any((1, 2)), active)
                np.testing.assert_allclose(peak(v-cv), peak(paths["anchor"]-cv), atol=1e-10, rtol=1e-8)
            scored = {k: parent.score(d, cv, ols, v, cuts) for k, v in paths.items()}
            ref0 = next(r for r in old["by_site"][held] if r["seed"] == seed)["metrics"]
            for k, old_k in (("anchor", "anchor"), ("prefix", "full_prefix"), ("learned_dynamics", "learned_dynamics")):
                if run.stable(scored[k]) != run.stable(ref0[old_k]):
                    raise ValueError("Frozen shape comparison changed")
            effect = lambda aa, bb: float(100.*parent.transfer.paired(paths[bb], paths[aa], y, valid)[1].sum()/mass)
            result[held].append(dict(seed=seed, scores=scored, zero_tangent_uses_anchor=nzero,
                direct=dict(dynamics_vs_tangent=effect("learned_dynamics", "tangent"), tangent_vs_anchor=effect("tangent", "anchor"),
                            tangent_vs_prefix=effect("tangent", "prefix"))))
        print(held, flush=True)
    draw = np.random.default_rng(817).integers(len(result), size=(3000, len(result)))
    flat = [r for v in result.values() for r in v]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    out = dict(tangent={f: sum(r["scores"]["tangent"][f] for r in flat) for f in ("edits", "risk_pass", "dual_positive", "views", "undefined")},
        zero_tangent_uses_anchor=sum(r["zero_tangent_uses_anchor"] for r in flat),
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]})
    run.once(study.OUT / "tangent_control.json", run.stable(dict(registration=key, analysis=run.artifact(p),
        code=run.artifact(Path(__file__)), summary=out, by_site=result, original_2pct=True,
        exact_rows_and_radii=True, no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
