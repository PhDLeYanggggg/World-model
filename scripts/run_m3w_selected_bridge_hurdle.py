"""Donor-only calibration of selected analytic bridges and easy support."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use the native arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_signed_bridge_risk as prior
from src.world_model import m3w_selected_bridge_hurdle as model

run, np, torch, transfer, source = prior.run, prior.np, prior.torch, prior.transfer, prior.source
OUT = prior.OUT.parent / "selected_bridge_hurdle_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, BATCH = 400, 128
GRID = [(0., 0.)]+[(s, e) for s in model.SAFETY for e in (0., *model.EASY)]
ARMS = ("utility", "safety_only", "support_hurdle", "hurdle_matched_independent", "tangent")


def register():
    key, frozen, _, _ = prior.register()
    p = prior.OUT / "analysis.json"
    a = run.read(p)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require the signed-risk result")
    f = OUT / "registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        previous=run.artifact(p), analytic_bank_only=True, native_steps=[8, 12], steps=STEPS, batch=BATCH,
        calibration_grid=GRID, calibration_only="out-of-fold complete selected policies on six donor sites",
        choose_by="mean donor dual-positive-pass fraction, pass fraction, lower utility, fewer edits",
        training_folds="full and six nested donor-held analytic heads per outer location and seed",
        no_new_official_split=True, original_2pct=True, confirmation_access=False, seeds=prior.prior.old.prior.SEEDS)))
    return run.sha(f), frozen


def training_data(d, bank, cut):
    cv, ols, b, x, scale, legal = bank
    t = model.targets(prior.prior.model.targets(cv, ols, b, d["y"], d["valid"], cut, scale))
    on = legal.any(1)
    return tuple(torch.from_numpy(v[on]) for v in (x, t, legal))


def fit(key, outer, seed, omit, sites, data, beat, replay):
    ident = dict(registration=key, outer=outer, seed=seed, omit=omit, fit=sites)
    p = PRIVATE / f"{outer}_{seed}_{omit or 'full'}.pt"
    torch.manual_seed(seed)
    net = model.Hurdle()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+144)
    step, trace = 0, []
    if p.exists():
        z = torch.load(p, weights_only=True, map_location="cpu")
        if z["identity"] != ident or z["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Unexpected analytic-head lineage")
        step, trace = z["step"], z["trace"]
        net.load_state_dict(z["net"])
        opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot fit")
    for t in range(step+1, STEPS+1):
        j = rng.integers(len(data), size=BATCH)
        pick = [rng.choice(len(v[0]), size=int((j == k).sum())) for k, v in enumerate(data)]
        x, y, legal = (torch.cat([data[k][i][at] for k, at in enumerate(pick)]) for i in range(3))
        loss = model.objective(net(x), y, legal)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=t, trace=trace, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(p)
            beat(phase="fit", outer=outer, omitted=omit, seed=seed, step=t)
    return net.eval(), dict(identity=ident, artifact=run.artifact(p), trace=trace), STEPS-step


@torch.inference_mode()
def predict(net, bank):
    _, _, _, x, scale, _ = bank
    v = net(torch.from_numpy(x))
    return np.concatenate((v[..., :2].numpy()*scale[:, None, None], v[..., 2:].sigmoid().numpy()), axis=2)


def selected(cv, bank, choice):
    path = cv.copy()
    at = choice >= 0
    path[at] = bank[at, choice[at]]
    np.testing.assert_array_equal((path != cv).any((1, 2)), at)
    return path


def score(d, cv, ols, path, cuts):
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    p = prior.prior.capacity.policy(path, cv, cv, y, valid, rec, easy_cut=cuts[0], hard_cut=cuts[1])
    rr = p["recordings"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    # Compute per-row gains once, including each missing step's debit.
    gain = np.stack([transfer.paired(b, path, y, valid)[1].sum(1) for b in (cv, ols)])
    risk = [name for name, row in rr.items() if row["supported"]]
    count = sum((gain[:, rec == k].sum(1) > 0.).all() for k in risk)
    return dict(vs_cv=float(100*gain[0].sum()/mass), vs_ols4=float(100*gain[1].sum()/mass), views=len(rr),
        dual_positive=int(count), risk_pass=len(risk), undefined=sum(v["completion_easy_risk_upper"] is None for v in rr.values()),
        edits=sum(v["selected"] for v in rr.values()), unknown_edits=sum(v["unknown_selected"] for v in rr.values()))


def select_grid(donor):
    table = []
    for i, (safe, easy) in enumerate(GRID):
        rows = [v[i] for v in donor.values()]
        table.append(dict(grid=[safe, easy], dual=float(np.mean([r["dual_positive"]/r["views"] for r in rows])),
            risk=float(np.mean([r["risk_pass"]/r["views"] for r in rows])),
            utility=float(np.mean([r["vs_cv"] for r in rows])), edits=sum(r["edits"] for r in rows)))
    def key(i):
        a = table[i]
        return a["dual"], a["risk"], a["utility"], -a["edits"]
    safe = max((i for i, (s, e) in enumerate(GRID) if s > 0. and e == 0.), key=key)
    hurdle = max((i for i, (_, e) in enumerate(GRID) if e > 0.), key=key)
    return (GRID[safe], GRID[hurdle]), table


def evaluate(d, bank, head, criteria, tangent, cuts):
    cv, ols, b, _, _, legal = bank
    pred = predict(head, bank)
    q = d["query"]
    paths = model.bank_decode(pred, legal, q, [GRID[0], *criteria])
    match = model.matched_independent(pred, legal, q, paths[-1])
    np.testing.assert_array_equal(np.bincount(q[match >= 0], minlength=q.max()+1),
                                  np.bincount(q[paths[-1] >= 0], minlength=q.max()+1))
    choices = dict(zip(ARMS[:-1], (*paths, match), strict=True))
    values = {k: selected(cv, b, choice) for k, choice in choices.items()}
    values["tangent"] = tangent
    scores = {k: score(d, cv, ols, a, cuts) for k, a in values.items()}
    valid, y = d["valid"], d["y"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    direct = {k: float(100*transfer.paired(values[k], values["support_hurdle"], y, valid)[1].sum()/mass)
              for k in ("utility", "safety_only", "hurdle_matched_independent", "tangent")}
    return dict(scores=scores, direct=direct, action_counts={k: np.bincount(a[a >= 0], minlength=3).tolist()
                                                           for k, a in choices.items()})


def summarize(sites):
    cells = [r for v in sites.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in sites.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    return dict(arms={k: dict(vs_cv=ci(lambda r: r["scores"][k]["vs_cv"]), vs_ols4=ci(lambda r: r["scores"][k]["vs_ols4"]),
        **{f: sum(r["scores"][k][f] for r in cells) for f in ("dual_positive", "risk_pass", "views", "undefined", "edits", "unknown_edits")}) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in cells[0]["direct"]},
        action_counts={k: np.sum([v["action_counts"][k] for v in cells], axis=0).tolist() for k in ARMS[:-1]})


def execute(key, frozen, beat, replay, cache):
    nets = transfer.frozen_models(*frozen[:5])
    data = {s: prior.prior.old.load_site(s) for s in source.SITES}
    banks = {s: prior.analytic(d) for s, d in data.items()}
    all_heads, results, steps = {}, {}, 0
    for j, outer in enumerate(source.SITES):
        donors = [s for s in source.SITES if s != outer]
        inner = {s: [k for k in donors if k != s] for s in donors}
        quantile = lambda ss: np.quantile(np.concatenate([data[s]["cut_samples"] for s in ss]), [.25, .75])
        cuts = quantile(donors)
        inside = {s: quantile(f) for s, f in inner.items()}
        tr = {s: [training_data(data[k], banks[k], inside[s][0]) for k in f] for s, f in inner.items()}
        full = [training_data(data[k], banks[k], cuts[0]) for k in donors]
        results[outer] = []
        for seed in prior.prior.old.prior.SEEDS:
            p = cache / f"{outer}_{seed}.json"
            if not p.exists():
                heads, scored = {}, {}
                for s in donors:
                    head, heads[s], n = fit(key, outer, seed, s, inner[s], tr[s], beat, replay)
                    steps += n
                    d, bank = data[s], banks[s]
                    cv, ols, b, _, _, legal = bank
                    paths = model.bank_decode(predict(head, bank), legal, d["query"], GRID)
                    scored[s] = [score(d, cv, ols, selected(cv, b, c), inside[s]) for c in paths]
                criteria, table = select_grid(scored)
                head, heads["full"], n = fit(key, outer, seed, None, donors, full, beat, replay)
                steps += n
                d = data[outer]
                a = prior.prior.prepare(d, transfer.forecast(d["h"], [nets[seed][j]]))
                results0 = evaluate(d, banks[outer], head, criteria, a["tangent"], cuts)
                run.once(p, run.stable(dict(registration=key, outer=outer, seed=seed, heads=heads,
                                           criteria=criteria, calibration=table, **results0)))
            a = run.read(p)
            if (a["registration"], a["outer"], a["seed"]) != (key, outer, seed):
                raise ValueError("Selected-policy checkpoint role changed")
            for s, h in a["heads"].items():
                run.checked(h["artifact"])
                all_heads[f"{outer}/{seed}/{s}"] = h
            results[outer].append({k: v for k, v in a.items() if k != "heads"})
            beat(phase="held", outer=outer, seed=seed)
    if len(all_heads) != 147:
        raise ValueError("Require full three-seed nested experiment")
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=all_heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(f), summary=summarize(results),
        by_site=results, original_2pct=True, no_held_selection=True, no_external_scoring=True, confirmation_access=False)))
    return steps


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

        reg = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                n = execute(*reg, beat, True, Path(t))
        else:
            n = execute(*reg, beat, False, PRIVATE)
        beat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
