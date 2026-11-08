"""Replace analytic-bridge risk with direct signed estimates and donor calibration."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native macOS arm64 .venv-pytorch interpreter")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_reference_bridge_policy as prior
from src.world_model import m3w_signed_bridge_risk as model

run, np, torch, source, transfer = prior.run, prior.np, prior.torch, prior.source, prior.transfer
OUT = prior.OUT.parent / "signed_bridge_risk_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, BATCH, QUANTILE = 400, 128, .60
ARMS = ("original", "direct", "donor_calibrated", "direct_matched_independent", "calibrated_matched_independent", "tangent")


def register():
    parent, frozen, _ = prior.register()
    paths = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "bottleneck")]
    a, h, b = [run.read(p) for p in paths]
    if (a["registration"] != parent or h["registration"] != parent or a["heads"] != run.sha(paths[1])
            or a["confirmation_access"] or b["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require the verified causal bridge fits")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), risk_model=run.artifact(Path(model.__file__)),
        parent=[run.artifact(p) for p in paths], only_opendd=True, horizon=[8, 12], steps=STEPS, batch=BATCH,
        calibration="six leave-one-donor-location-out fits; analytic paths only; equal-location residual offsets",
        residual_quantile=QUANTILE, both_risks_signed=True, hold_gain_and_neural_risk_fixed=True,
        primary="direct and donor-calibrated bridge risk versus original at original recording rule",
        original_2pct=True, seeds=prior.old.prior.SEEDS, threads=4, workers=0, confirmation_access=False)))
    return run.sha(f), frozen, a, h


def analytic(d):
    h = d["h"]
    cv, ols, _, _ = transfer.motion.rollout(h, 12)
    step = (ols-cv)[:, None]
    bank = cv[:, None]+step*np.asarray(prior.capacity.MIX)[None, :, None, None]
    bank[:, -1] = ols
    x, scale, legal, _ = prior.model.features(h, cv, ols, bank)
    return cv, ols, bank, x, scale, legal


def training_data(d, analytic_bank, cut):
    cv, ols, bank, x, scale, legal = analytic_bank
    target = model.signed_target(prior.model.targets(cv, ols, bank, d["y"], d["valid"], cut, scale))
    on = legal.any(1)
    return tuple(torch.from_numpy(z[on]) for z in (x, target, legal))


def fit(key, held, seed, omitted, sites, data, beat, replay):
    ident = dict(registration=key, held=held, seed=seed, omitted=omitted, fit=sites)
    path = PRIVATE / f"{held}_{seed}_{omitted or 'full'}.pt"
    torch.manual_seed(seed)
    net = model.SignedRisk()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+113)
    step, trace = 0, []
    if path.exists():
        z = torch.load(path, weights_only=True, map_location="cpu")
        if z["identity"] != ident or z["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Wrong signed-risk checkpoint role")
        step, trace = z["step"], z["trace"]
        net.load_state_dict(z["net"])
        opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]
    if replay and step != STEPS:
        raise ValueError("Replay may not fit")
    for t in range(step+1, STEPS+1):
        g = rng.integers(len(data), size=BATCH)
        pick = [rng.choice(len(v[0]), size=int((g == j).sum())) for j, v in enumerate(data)]
        x, y, mask = (torch.cat([data[j][k][at] for j, at in enumerate(pick)]) for k in range(3))
        p = net(x)[mask]
        loss = (torch.nn.functional.smooth_l1_loss(p, y[mask], reduction="none", beta=.01)
                *p.new_tensor([1., 2.])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, net=net.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state,
                            step=t, trace=trace), tmp)
            tmp.replace(path)
            beat(phase="fit", held=held, seed=seed, omitted=omitted, step=t)
    return net.eval(), dict(identity=ident, artifact=run.artifact(path), loss=trace), STEPS-step


@torch.inference_mode()
def evaluate(d, p, original, head, correction, cut, previous):
    cv, ols, bank = (p[k] for k in ("cv", "ols", "bank"))
    z = original(torch.from_numpy(p["x"])).numpy()*p["scale"][:, None, None]
    u = head(torch.from_numpy(p["x"][:, -3:])).numpy()
    direct = model.replace_bridge_risk(z, u, p["scale"], np.zeros((3, 2)))
    cal = model.replace_bridge_risk(z, u, p["scale"], correction)
    decode = lambda a: prior.model.query_choices(a, p["legal"], d["query"], p["motion"])
    first, second, third = [decode(a) for a in (z, direct, cal)]
    actions = dict(original=first["joint"], direct=second["joint"], donor_calibrated=third["joint"],
                   direct_matched_independent=second["matched_independent"],
                   calibrated_matched_independent=third["matched_independent"])
    q = d["query"]
    for arm, match in (("direct", "direct_matched_independent"), ("donor_calibrated", "calibrated_matched_independent")):
        np.testing.assert_array_equal(np.bincount(q[actions[arm] >= 0], minlength=q.max()+1),
                                      np.bincount(q[actions[match] >= 0], minlength=q.max()+1))
    values = {"tangent": p["tangent"]}
    for k, a in actions.items():
        on = a >= 0
        path = cv.copy()
        path[on] = bank[on, a[on]]
        np.testing.assert_array_equal((path != cv).any((1, 2)), on)
        values[k] = path
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()

    def effect(a, b):
        return float(100*transfer.paired(b, a, y, valid)[1].sum()/mass)

    result = {}
    for k, a in values.items():
        p2 = prior.capacity.policy(a, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
        pass_rec = [s for s, row in p2.items() if row["supported"]]
        both = 0
        for s in pass_rec:
            at = rec == s
            both += all(transfer.paired(b[at], a[at], y[at], valid[at])[1].sum() > 0. for b in (cv, ols))
        result[k] = dict(vs_cv=effect(a, cv), vs_ols4=effect(a, ols), dual_positive_passes=both,
            risk_passes=len(pass_rec), views=len(p2), undefined=sum(v["completion_easy_risk_upper"] is None for v in p2.values()),
            edits=int((a != cv).any((1, 2)).sum()), unknown_edits=sum(v["unknown_selected"] for v in p2.values()),
            counts=np.bincount(actions[k][actions[k] >= 0], minlength=bank.shape[1]).tolist() if k in actions else None)
    if any(result["original"][k] != previous["arms"]["joint"][k] for k in result["original"] if k != "counts"):
        raise ValueError("Original bridge decoder failed exact score replay")
    names = (("direct", "original"), ("donor_calibrated", "original"), ("donor_calibrated", "direct"),
             ("direct", "direct_matched_independent"), ("donor_calibrated", "calibrated_matched_independent"),
             ("donor_calibrated", "tangent"), ("direct", "tangent"))
    return dict(arms=result, direct={f"{a}_vs_{b}": effect(values[a], values[b]) for a, b in names})


def summary(sites):
    flat = [r for v in sites.values() for r in v]
    draws = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in sites.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    arms = {}
    for k in ARMS:
        rows = [v["arms"][k] for v in flat]
        arms[k] = dict(vs_cv=ci(lambda r: r["arms"][k]["vs_cv"]), vs_ols4=ci(lambda r: r["arms"][k]["vs_ols4"]),
            **{s: sum(r[s] for r in rows) for s in ("dual_positive_passes", "risk_passes", "views", "undefined", "edits", "unknown_edits")},
            counts=np.sum([r["counts"] for r in rows], axis=0).tolist() if rows[0]["counts"] is not None else None)
    return dict(arms=arms, direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]})


def execute(key, frozen, a, h, beat, replay, cache):
    nets = transfer.frozen_models(*frozen[:5])
    data = {s: prior.old.load_site(s) for s in source.SITES}
    analytic_banks = {s: analytic(d) for s, d in data.items()}
    results, heads, updates = {}, {}, 0
    for j, held in enumerate(source.SITES):
        donors = [s for s in source.SITES if s != held]
        outer_cut = np.quantile(np.concatenate([data[s]["cut_samples"] for s in donors]), [.25, .75])
        sites = {s: [v for v in donors if v != s] for s in donors}
        inner_cut = {s: np.quantile(np.concatenate([data[k]["cut_samples"] for k in fit]), .25) for s, fit in sites.items()}
        pools = {s: [training_data(data[k], analytic_banks[k], inner_cut[s]) for k in sites[s]] for s in donors}
        full = [training_data(data[s], analytic_banks[s], outer_cut[0]) for s in donors]
        results[held] = []
        for i, seed in enumerate(prior.old.prior.SEEDS):
            file = cache / f"{held}_{seed}.json"
            if not file.exists():
                residuals, stamps = [], {}
                for s in donors:
                    net, stamps[s], n = fit(key, held, seed, s, sites[s], pools[s], beat, replay)
                    updates += n
                    x, y, legal = training_data(data[s], analytic_banks[s], inner_cut[s])
                    with torch.inference_mode():
                        residuals.append((y.numpy()-net(x).numpy(), legal.numpy()))
                correction = model.location_correction(residuals, QUANTILE)
                net, stamps["full"], n = fit(key, held, seed, None, donors, full, beat, replay)
                updates += n
                receipt = h["heads"][f"{held}/{seed}"]
                z = torch.load(run.checked(receipt["checkpoint"]), weights_only=True, map_location="cpu")
                if run.stable(z["identity"]) != receipt["identity"] or z["step"] != prior.STEPS:
                    raise ValueError("Frozen gain checkpoint differs from prior experiment")
                original = prior.model.BridgeValue()
                original.load_state_dict(z["net"])
                d = data[held]
                pred = prior.prepare(d, transfer.forecast(d["h"], [nets[seed][j]]))
                r = evaluate(d, pred, original.eval(), net, correction, outer_cut, a["by_site"][held][i])
                run.once(file, run.stable(dict(registration=key, held=held, seed=seed, correction=correction.tolist(),
                                               heads=stamps, **r)))
            r = run.read(file)
            if (r["registration"], r["held"], r["seed"]) != (key, held, seed):
                raise ValueError("Scoring fold role changed")
            for s, m in r["heads"].items():
                run.checked(m["artifact"])
                heads[f"{held}/{seed}/{s}"] = m
            results[held].append({k: v for k, v in r.items() if k != "heads"})
            beat(phase="held", held=held, seed=seed)
    if len(heads) != 147:
        raise ValueError("Incomplete signed-risk head roster")
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=summary(results), by_site=results,
        calibration_is_development_only=True, no_held_tuning=True, original_2pct=True, confirmation_access=False, deployment=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        args0 = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                updates = execute(*args0, beat, True, Path(t))
        else:
            updates = execute(*args0, beat, False, PRIVATE)
        beat(phase="complete", updates=updates, replay=args.replay)


if __name__ == "__main__":
    main()
