"""Isolate candidate geometry at identical intervention rows, times and radii."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_recording_swap_rank as prior
from src.evaluation import m3w_fixed_roster_shape_capacity as capacity
from src.world_model import m3w_radius_matched_turn_menu as model

base, parent, run, torch, study = prior.base, prior.parent, prior.run, prior.torch, prior.study
OUT = prior.OUT.parent / "radius_matched_turn_menu_v1"
ARMS = ("anchor", "timing_prefix", "full_prefix", "full_consensus", "cv_gain_oracle")


def register():
    key, a, ph, p, h, *_ = prior.register()
    path = prior.OUT / "analysis.json"
    f = run.read(path)
    if f["registration"] != key or f["confirmation_access"] or not f["original_2pct"]:
        raise ValueError("Require bound fixed-budget rank controls")
    out = OUT / "registration.json"
    run.once(out, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        capacity=run.artifact(Path(capacity.__file__)), parent=run.artifact(path), native_steps=[8, 12], arms=ARMS,
        shapes=model.NAMES, radius="exact frozen CV-relative peak for each realized row", exact_row_count=True,
        original_2pct=True, no_training=True, confirmation_access=False,
        shape_selector="mean observed-prefix score, or agreement in all three prefixes; no held tuning",
        oracles="label-aware only; same active rows and radii for both timing and expanded bank")))
    return run.sha(out), a, ph, p, h


def audit(d, cv, ols, b, use, cuts):
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    cols = dict(timing=Counter(), full=Counter())
    transitions = Counter()
    for name in sorted(set(rec)):
        ix = rec == name
        t = capacity.assess(cv[ix], ols[ix], b[ix, :3], y[ix], valid[ix], name, cuts, use[ix])
        f = capacity.assess(cv[ix], ols[ix], b[ix], y[ix], valid[ix], name, cuts, use[ix])
        if t == "verified_positive_both" and f not in ("verified_positive_both", "unresolved"):
            raise ValueError("Expanded menu lost a verified witness from the subset")
        cols["timing"][t] += 1
        cols["full"][f] += 1
        transitions[f"{t} -> {f}"] += 1
    return dict(**cols, transitions=transitions)


def score(d, bank, menu, count, cuts):
    cv, ols, b = bank[:3]
    choose = np.arange(2) < count[:, None]
    active, amp = np.zeros(len(cv), dtype=bool), np.zeros(len(cv))
    row, arm = menu["row"][choose], menu["arm"][choose]
    active[row], amp[row] = True, np.array([.25, .5, 1.])[arm]
    c, o, paths = model.make_menu(d["h"], 12, amp)
    np.testing.assert_array_equal(cv, c)
    np.testing.assert_array_equal(ols, o)
    np.testing.assert_array_equal(paths[row, 0], b[row, arm])
    probes = model.prefix_rank(d["h"], amp, active)
    rank = probes.mean(0)
    agree = (probes > probes[..., :1]).all(0)
    agree[:, 0] = True
    y, valid = d["y"], d["valid"]
    values = np.stack([parent.transfer.paired(cv, a, y, valid)[1].sum(1) for a in paths.swapaxes(0, 1)], axis=1)
    selections = dict(anchor=np.zeros(len(cv), dtype=int), timing_prefix=rank[:, :3].argmax(1),
        full_prefix=rank.argmax(1), full_consensus=np.where(agree, rank, -np.inf).argmax(1), cv_gain_oracle=values.argmax(1))
    preds, metrics, hist = {}, {}, {}
    radius = model.peak(paths[:, 0]-cv)
    for k, a in selections.items():
        pred = paths[np.arange(len(cv)), a]
        np.testing.assert_array_equal((pred != cv).any((1, 2)), active)
        np.testing.assert_allclose(model.peak(pred-cv), radius, atol=1e-10, rtol=1e-8)
        preds[k] = pred
        metrics[k] = parent.score(d, cv, ols, pred, cuts)
        hist[k] = np.bincount(a[active], minlength=len(model.NAMES)).tolist()
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = tuple((k, "anchor") for k in ARMS[1:])+(("full_prefix", "timing_prefix"), ("full_consensus", "timing_prefix"))
    direct = {f"{k}_vs_{r}": float(100.*parent.transfer.paired(preds[r], preds[k], y, valid)[1].sum()/mass) for k, r in pairs}
    return dict(metrics=metrics, shapes=hist, direct=direct, capacity=audit(d, cv, ols, paths, active, cuts))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, p, h = register()
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    result = {}
    for held, d in sites.items():
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        cuts = np.quantile(np.concatenate([v["cut_samples"] for s, v in sites.items() if s != held]), [.25, .75])
        b = parent.prior.analytic(d)
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            m = study.menus(d, b, rk)
            x, _, _ = base.features(d, b, m)
            net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                z = net(torch.from_numpy(x["motion"])).numpy()
            c = study.model.streaming_count(base.model.score(z, risk=False), m["legal"], rec, starts)
            val = score(d, b, m, c, cuts)
            old = next(v for v in p["by_site"][held] if v["seed"] == seed)
            if run.stable(val["metrics"]["anchor"]) != run.stable(old["online"]["motion_gain_only"]):
                raise ValueError("Frozen gain-only roster or paths changed")
            result[held].append(dict(seed=seed, **val))
        print(held, flush=True)
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    cap = {k: Counter() for k in ("timing", "full", "transitions")}
    for r in flat:
        for k in cap:
            cap[k].update(r["capacity"][k])
    out = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
            **{f: sum(r["metrics"][k][f] for r in flat) for f in ("edits", "views", "risk_pass", "dual_positive", "undefined")},
            shapes=np.sum([r["shapes"][k] for r in flat], axis=0).tolist()) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]}, capacity=cap)
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=out, by_site=result,
        original_2pct=True, no_training=True, confirmation_access=False, oracle_label_aware_only=True)))


if __name__ == "__main__":
    main()
