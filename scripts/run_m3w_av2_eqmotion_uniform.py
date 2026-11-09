"""Match the edit budget with one trajectory family per scenario."""

import argparse
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_eqmotion as run

prev, av, np, torch, study, base = run.prev, run.av, run.np, run.torch, run.study, run.base


def uniform_opts(opts, family, slots):
    if slots == 1:
        return opts
    return opts[family[opts[:, 0]] == family[opts[:, 1]]]


@torch.inference_mode()
def predict(d, bank, radii, models):
    c, o = d["cv"], d["ols4"]
    if not len(radii):
        return c
    paths, row, col, opts, group = av.optional.options(c, bank, radii)
    opts = uniform_opts(opts[group == (2 if len(radii) == 2 else 0)], col % bank.shape[1], len(radii))
    if not len(opts):
        raise ValueError("The original OLS4 assignment should remain in the uniform menu")
    x, unit, legal, _ = av.frozen.shared.costs.features(d["h"], c, o, paths)
    if not legal[row, col].all():
        raise ValueError("Collapsed exact-slot action")
    inp = torch.from_numpy(x[row, col])
    z = torch.stack([net(inp) for net in models]).mean(0).numpy()*unit[row, None]
    value, _ = av.turn.refit.model.ranks(z, opts)
    path = av.optional.paint(c, paths, row, col, opts[int(value.argmax())])
    r = av.motion.peak(path-c)
    np.testing.assert_allclose(np.sort(r[r > 0.]), np.sort(radii), atol=1e-8, rtol=1e-8)
    return path


def setup():
    parent, previous, source, _, costs, _ = run.register()
    p = run.OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != parent or f["held_labels_used"] or f["calibration_labels_used"]:
        raise ValueError("Require frozen forecaster before scoring")
    reg = run.OUT/"uniform_registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), code=study.artifact(Path(__file__)),
        policy="same expanded menu and additive scores, allow only common candidate family across two slots",
        roles="same city folds, exact same peaks and candidate availability",
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), previous, source, costs, f


def freeze(key, data, costs, f):
    weights = {s: av.frozen.load(s, costs, av.turn.refit.model.CostHead, av.turn.refit.STEPS) for s in run.SEEDS}
    receipts = {}
    for name, z in f["folds"].items():
        seed = z["identity"]["seed"]
        with np.load(study.checked(z["forecast"]), allow_pickle=False) as raw:
            p = dict(raw)
        out = []
        for n, lo, hi in zip(p["names"], p["offsets"][:-1], p["offsets"][1:], strict=True):
            d = data[n]
            r = run.radii(d, seed)
            q = predict(d, run.expanded(d, p["raw_eq"][lo:hi]), r, weights[seed])
            if len(r) < 2:
                np.testing.assert_array_equal(q, p["eq_additive"][lo:hi])
            out.append(q)
        receipts[name] = av.turn.exp.previous.save_bank(run.LOCAL/f"{name}_uniform.npz",
            dict(registration=key, names=p["names"], offsets=p["offsets"], path=np.concatenate(out)))
        run.beat("uniform_freeze", fold=name, done=len(receipts), total=len(f["folds"]))
    base.immutable(run.OUT/"uniform_freeze.json", dict(registration=key, forecasts=receipts, future_used=False))


def score(key, data, source, f):
    path = run.OUT/"uniform_freeze.json"
    seal = study.read(path)
    if seal["registration"] != key or seal["future_used"] or set(seal["forecasts"]) != set(f["folds"]):
        raise ValueError("All scene-uniform forecasts must precede scoring")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows = []
    for name, z in f["folds"].items():
        city, seed = z["identity"]["held"], z["identity"]["seed"]
        with (np.load(study.checked(z["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(seal["forecasts"][name]), allow_pickle=False) as other):
            np.testing.assert_array_equal(raw["names"], other["names"])
            np.testing.assert_array_equal(raw["offsets"], other["offsets"])
            if other["registration"].item() != key:
                raise ValueError("Wrong scene-uniform forecast lineage")
            for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                d, (y, v) = data[n], truth[n]
                u = other["path"][lo:hi]
                metric = prev.metrics(u, d, y, v, z["cut"])
                pair = {a: av.frontier.pair_reduction(raw[a][None, lo:hi], u, y, v)[0].tolist()
                        for a in ("eq_additive", "eq_joint")}
                rows.append(dict(clip=str(n), city=city, seed=seed, metric=metric, pair=pair))
        run.beat("uniform_score", fold=name, views=len(rows))
    edited = [r["metric"]["risk"] for r in rows if r["metric"]["risk"]["selected"]]
    result = dict(registration=key, freeze=study.artifact(path), result_source="fresh_run",
        edits=sum(x["selected"] for x in edited), edited=len(edited), risk_pass=sum(x["supported"] for x in edited),
        dual_risk_pass=sum(run.ceiling.eligible_additive(r["metric"]) for r in rows),
        easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in edited),
        easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in edited),
        metrics={k: av.interval(rows, lambda r: r["metric"][k]) for k in ("ade", "fde", "gain", "hard", "tail")},
        paired_gain_vs_uniform={a: av.interval(rows, lambda r: r["pair"][a]) for a in ("eq_additive", "eq_joint")},
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"uniform_summary.json", result)
    print(study.json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with (run.LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, previous, source, costs, f = setup()
        data = run.scale.read_inputs(previous, source)
        if phase == "freeze":
            freeze(key, data, costs, f)
        else:
            score(key, data, source, f)
        run.beat("uniform_"+phase+"_complete")


if __name__ == "__main__":
    main()
