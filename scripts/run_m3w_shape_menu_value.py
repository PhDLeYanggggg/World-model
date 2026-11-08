"""Compare learned shape choices with the frozen observed-prefix heuristic."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_radius_matched_turn_menu as prior
from src.world_model import m3w_shape_menu_value as model

base, parent, run, torch, study = prior.base, prior.parent, prior.run, prior.torch, prior.study
OUT = prior.OUT.parent / "shape_menu_value_v1"
PRIVATE = prior.prior.PRIVATE.parent / OUT.name
STEPS = 800
ARMS = ("anchor", "timing_prefix", "full_prefix", "learned_gain", "learned_risk", "learned_dynamics")


def register():
    key, a, ph, p, h = prior.register()
    file = prior.OUT / "analysis.json"
    prev = run.read(file)
    if prev["registration"] != key or prev["confirmation_access"] or not prev["oracle_label_aware_only"]:
        raise ValueError("Require bound equal-radius candidate geometry")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        prior=run.artifact(file), arms=ARMS, steps=STEPS, seeds=study.prior.SEEDS, native_steps=[8, 12],
        training="all fitting rows with three fixed bridge amplitudes; no nominated rows or outcomes used for eligibility",
        comparison="learned value/risk and learned motion-proxy shape choice versus full observed-prefix; frozen interventions",
        radius="exact per selected row", no_threshold_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, prev


def prepare(d, cut):
    cv, ols, _, _ = model.menu.rollout(d["h"], 12)
    on = model.menu.peak(ols-cv) > 0.
    out = []
    for alpha in (.25, .5, 1.):
        h, y, valid = (d[k][on] for k in ("h", "y", "valid"))
        x, cv, _, b, _, rot, scale = model.features(h, np.full(len(h), alpha))
        v, m = model.targets(cv, b, y, valid, cut, rot, scale)
        out.append((x, v, m, valid))
    return tuple(torch.from_numpy(np.concatenate([z[i] for z in out])) for i in range(4))


def batches(data):
    sites = [d for d in data if len(d[0])]

    def batch(rng):
        g = rng.integers(len(sites), size=128)
        at = [rng.integers(len(x[0]), size=int((g == j).sum())) for j, x in enumerate(sites)]
        return tuple(torch.cat([x[i][idx] for x, idx in zip(sites, at, strict=True)]) for i in range(4))

    return batch


@torch.inference_mode()
def evaluate(d, bank, m, count, nets, cuts):
    cv, ols, b = bank[:3]
    take = np.arange(2) < count[:, None]
    rows, arm = m["row"][take], m["arm"][take]
    active = np.zeros(len(cv), dtype=bool)
    active[rows] = True
    y, valid = d["y"], d["valid"]
    hist = {k: [0]*len(model.menu.NAMES) for k in ARMS}
    if len(rows):
        amp = np.array([.25, .5, 1.])[arm]
        x, cc, oo, paths, geom, _, _ = model.features(d["h"][rows], amp)
        np.testing.assert_array_equal(cv[rows], cc)
        np.testing.assert_array_equal(ols[rows], oo)
        np.testing.assert_array_equal(b[rows, arm], paths[:, 0])
        probe = model.menu.prefix_rank(d["h"][rows], amp, np.ones(len(rows), dtype=bool)).mean(0)
        z = torch.from_numpy(x)
        choices = dict(anchor=np.zeros(len(rows), dtype=int), timing_prefix=probe[:, :3].argmax(1), full_prefix=probe.argmax(1),
                       **model.choose(nets["value"](z).numpy(), nets["dynamics"](z).numpy(), geom))
    pred, summary = {}, {}
    for k in ARMS:
        a = cv.copy()
        if len(rows):
            a[rows] = paths[np.arange(len(rows)), choices[k]]
            hist[k] = np.bincount(choices[k], minlength=len(model.menu.NAMES)).tolist()
            np.testing.assert_allclose(model.menu.peak(a[rows]-cv[rows]), model.menu.peak(paths[:, 0]-cv[rows]), atol=1e-10, rtol=1e-8)
        np.testing.assert_array_equal((a != cv).any((1, 2)), active)
        pred[k] = a
        summary[k] = parent.score(d, cv, ols, a, cuts)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "full_prefix") for k in ARMS[3:]]+[(k, "anchor") for k in ARMS[3:]]
    direct = {f"{k}_vs_{ref}": float(100.*parent.transfer.paired(pred[ref], pred[k], y, valid)[1].sum()/mass) for k, ref in pairs}
    return dict(metrics=summary, shapes=hist, direct=direct)


def execute(key, a, ph, h, prev, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    rows, frozen, steps = {}, {}, 0
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[k]["cut_samples"] for k in donors]), [.25, .75])
        train = [prepare(sites[s], cuts[0]) for s in donors]
        width = train[0][0].shape[1]
        batch = batches(train)
        b = parent.prior.analytic(d)
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            nets = {}
            for kind in ("value", "dynamics"):
                name = f"{held}_{seed}_{kind}"
                identity = dict(registration=key, held=held, donors=donors, seed=seed, kind=kind)
                nets[kind], frozen[name], u = study.fit(identity, name, seed, lambda: model.Head(width, kind), STEPS, batch,
                    model.objective, replay, beat)
                steps += u
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            m = study.menus(d, b, rk)
            x, _, _ = base.features(d, b, m)
            net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                pred = net(torch.from_numpy(x["motion"])).numpy()
            c = study.model.streaming_count(base.model.score(pred, risk=False), m["legal"], rec, starts)
            v = evaluate(d, b, m, c, nets, cuts)
            old = next(z for z in prev["by_site"][held] if z["seed"] == seed)
            for k in ARMS[:3]:
                if run.stable(v["metrics"][k]) != run.stable(old["metrics"][k]):
                    raise ValueError("Frozen same-radius controls did not reproduce")
            rows[held].append(dict(seed=seed, **v))
            beat(phase="held", held=held, seed=seed)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    flat = [r for v in rows.values() for r in v]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    out = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
        **{f: sum(r["metrics"][k][f] for r in flat) for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")},
        shapes=np.sum([r["shapes"][k] for r in flat], axis=0).tolist()) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]})
    hfile = OUT / "head_freeze.json"
    run.once(hfile, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(hfile), summary=out, by_site=rows,
        exact_rows_and_radii=True, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        u = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=u, replay=args.replay)


if __name__ == "__main__":
    main()
