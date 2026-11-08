"""Fit relative swap scores without changing the frozen intervention budget."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_selected_bridge_hurdle as parent
from src.world_model import m3w_matched_bridge_swaps as model

run, np, torch = parent.run, parent.np, parent.torch
OUT = parent.OUT.parent / "matched_bridge_swaps_v1"
PRIVATE = parent.PRIVATE.parent / OUT.name
STEPS, BATCH = 700, 96
MODES = {"joint_point": (True, False), "additive_rank": (False, True), "joint_rank": (True, True)}
CONTROLS = ("independent", "frozen_hurdle")
SEEDS = parent.prior.prior.old.prior.SEEDS


def register():
    old_key, _ = parent.register()
    paths = [parent.OUT / f"{x}.json" for x in ("analysis", "head_freeze", "exact_analytic_capacity")]
    a, h, c = [run.read(p) for p in paths]
    if (a["registration"] != old_key or a["heads"] != run.sha(paths[1]) or c["analysis"]["sha256"] != run.sha(paths[0])
            or h["registration"] != old_key or any(v["confirmation_access"] or not v["original_2pct"] for v in (a, c))):
        raise ValueError("Require verified frozen count policy")
    p = OUT / "registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=[run.artifact(p) for p in paths], seeds=SEEDS, updates=STEPS, batch=BATCH, steps=[8, 12],
        comparison="joint_rank vs frozen same-count independent; joint point and additive rank are fixed ablations",
        training="sample two legal configurations at same nonzero count; fit excludes outer location",
        counts="frozen donor-selected hurdle per outer location and seed; no held selection", original_2pct=True,
        no_held_tuning=True, threads=4, workers=0, confirmation_access=False)))
    return run.sha(p), a, h


def groups(d):
    ids, start, size = np.unique(d["query"], return_index=True, return_counts=True)
    for i, s, n in zip(ids, start, size, strict=True):
        ix = slice(s, s+n)
        if not (d["query"][ix] == i).all():
            raise ValueError("Expected contiguous query groups")
        yield ix, int(n)


def prepare(d, bank, cut):
    cv, ols, b, x, scale, legal = bank
    parts, offset = model.label_parts(cv, ols, b, d["y"], d["valid"], cut)
    table = []
    for ix, n in groups(d):
        for k in (1, 2):
            at = model.choices(legal[ix], k)
            if len(at) > 1:
                target = model.utility(parts[ix], at, offset[ix].sum(), 12.*float(np.median(scale[ix])))
                absolute = np.where(at >= 0, at+ix.start*b.shape[1], -1)
                table.append((absolute, target.astype(np.float32), np.float32(np.log1p(n))))
    return x.reshape(-1, x.shape[-1]), table


def batch(data, rng):
    feat, target, mask, size = [], [], [], []
    for j in rng.integers(len(data), size=BATCH):
        x, table = data[j]
        actions, value, n = table[rng.integers(len(table))]
        a = rng.integers(len(actions))
        b = rng.integers(len(actions)-1)
        if b >= a:
            b += 1
        pick = actions[[a, b]]
        feat.append(model.gather(x, pick))
        target.append(value[[a, b]])
        mask.append(pick[:, 1] >= 0)
        size.append([n, n])
    return tuple(torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (feat, target, mask, size))


def fit(key, held, seed, mode, donors, data, heartbeat, replay):
    identity = dict(registration=key, held=held, seed=seed, mode=mode, donors=donors)
    p = PRIVATE / f"{held}_{seed}_{mode}.pt"
    torch.manual_seed(seed)
    joint, rank = MODES[mode]
    net = model.SwapValue(joint)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+391)
    step, loss_trace = 0, []
    if p.exists():
        z = torch.load(p, weights_only=True, map_location="cpu")
        if z["identity"] != identity or z["step"] not in range(100, STEPS+1, 100):
            raise ValueError("Swap checkpoint role changed")
        step, loss_trace = z["step"], z["loss"]
        net.load_state_dict(z["net"])
        opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]
    if replay and step != STEPS:
        raise ValueError("Replay cannot train")
    for t in range(step+1, STEPS+1):
        x, y, pair, n = batch(data, rng)
        loss = model.objective(net(x, pair, n), y, rank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            loss_trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, step=t, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=loss_trace), tmp)
            tmp.replace(p)
            heartbeat(phase="fit", held=held, seed=seed, mode=mode, step=t)
    return net.eval(), dict(identity=identity, artifact=run.artifact(p), loss=loss_trace), STEPS-step


@torch.inference_mode()
def decode(d, x, legal, fixed, nets):
    opts = []
    for ix, n in groups(d):
        count = int((fixed[ix] >= 0).sum())
        if count:
            a = model.choices(legal[ix], count)
            if not len(a):
                raise ValueError("Frozen count has no feasible assignment")
            at = np.where(a >= 0, a+ix.start*legal.shape[1], -1)
            opts.append(at)
            if not np.isfinite(x[ix]).all():
                raise ValueError("Nonfinite causal action features")
        else:
            opts.append(np.empty((0, 2), dtype=np.int64))
    options = np.concatenate(opts)
    if not len(options):
        return {name: np.full_like(fixed, -1) for name in nets}
    sizes = np.repeat([np.log1p(n) for _, n in groups(d)], [len(a) for a in opts]).astype(np.float32)
    pred = {name: [] for name in nets}
    flat = x.reshape(-1, x.shape[-1])
    for start in range(0, len(options), 8192):
        a = options[start:start+8192]
        f = torch.from_numpy(model.gather(flat, a).astype(np.float32, copy=False))
        p = torch.from_numpy((a[:, 1] >= 0).astype(np.float32))
        n = torch.from_numpy(sizes[start:start+8192])
        for name, net in nets.items():
            pred[name].append(net(f, p, n).numpy())
    result = {}
    for name, vals in pred.items():
        vals = np.concatenate(vals)
        if not np.isfinite(vals).all():
            raise ValueError("Nonfinite swap scores")
        out, pos = np.full_like(fixed, -1), 0
        for a in opts:
            if len(a):
                picked = a[vals[pos:pos+len(a)].argmax()]
                row, arm = np.divmod(picked[picked >= 0], legal.shape[1])
                out[row] = arm
            pos += len(a)
        result[name] = out
    return result


def evaluate(d, bank, head, criteria, cuts, nets):
    cv, ols, b, x, _, legal = bank
    pred = parent.predict(head, bank)
    q, y, valid = (d[k] for k in ("query", "y", "valid"))
    fixed = parent.model.bank_decode(pred, legal, q, [criteria])[0]
    independent = parent.model.matched_independent(pred, legal, q, fixed)
    choices = dict(independent=independent, frozen_hurdle=fixed, **decode(d, x, legal, fixed, nets))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    old = parent.selected(cv, b, independent)
    scores, direct, changes = {}, {}, {}
    count = np.bincount(q[fixed >= 0], minlength=q.max()+1)
    for k, a in choices.items():
        np.testing.assert_array_equal(np.bincount(q[a >= 0], minlength=q.max()+1), count)
        path = parent.selected(cv, b, a)
        scores[k] = parent.score(d, cv, ols, path, cuts)
        direct[k] = float(100*parent.transfer.paired(old, path, y, valid)[1].sum()/mass)
        changes[k] = dict(recipients=int(((a >= 0) & (independent < 0)).sum()),
                          amplitude_only=int(((a >= 0) & (independent >= 0) & (a != independent)).sum()))
    return dict(scores=scores, direct_vs_independent=direct, changes=changes)


def summary(results):
    draw = np.random.default_rng(817).integers(len(results), size=(3000, len(results)))
    rows = [r for rr in results.values() for r in rr]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in results.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    return {k: dict(vs_cv=ci(lambda r: r["scores"][k]["vs_cv"]),
        vs_ols4=ci(lambda r: r["scores"][k]["vs_ols4"]), direct_vs_independent=ci(lambda r: r["direct_vs_independent"][k]),
        **{f: sum(r["scores"][k][f] for r in rows) for f in ("dual_positive", "risk_pass", "views", "undefined", "edits", "unknown_edits")},
        **{f: sum(r["changes"][k][f] for r in rows) for f in ("recipients", "amplitude_only")}) for k in (*CONTROLS, *MODES)}


def execute(key, a, h, heartbeat, replay, cache):
    data = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(d) for s, d in data.items()}
    results, checkpoints, steps = {}, {}, 0
    for held in data:
        donors = [s for s in data if s != held]
        cuts = np.quantile(np.concatenate([data[s]["cut_samples"] for s in donors]), [.25, .75])
        train = [prepare(data[s], banks[s], cuts[0]) for s in donors]
        results[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            file = cache / f"{held}_{seed}.json"
            if not file.exists():
                head = parent.model.Hurdle()
                meta = h["heads"][f"{held}/{seed}/full"]
                z = torch.load(run.checked(meta["artifact"]), weights_only=True, map_location="cpu")
                if z["identity"] != meta["identity"] or z["step"] != parent.STEPS:
                    raise ValueError("Unexpected count-policy checkpoint")
                head.load_state_dict(z["net"])
                nets, saved = {}, {}
                for mode in MODES:
                    nets[mode], saved[mode], n = fit(key, held, seed, mode, donors, train, heartbeat, replay)
                    steps += n
                values = evaluate(data[held], banks[held], head.eval(), cell["criteria"][1], cuts, nets)
                for name, parent_name in (("independent", "hurdle_matched_independent"), ("frozen_hurdle", "support_hurdle")):
                    if run.stable(values["scores"][name]) != run.stable(cell["scores"][parent_name]):
                        raise ValueError("Frozen matched comparison did not reproduce")
                run.once(file, run.stable(dict(registration=key, held=held, seed=seed, checkpoints=saved, **values)))
            r = run.read(file)
            if (r["registration"], r["held"], r["seed"]) != (key, held, seed):
                raise ValueError("Swap result lineage changed")
            for mode, v in r["checkpoints"].items():
                run.checked(v["artifact"])
                checkpoints[f"{held}/{seed}/{mode}"] = v
            results[held].append({k: v for k, v in r.items() if k != "checkpoints"})
            heartbeat(phase="held", held=held, seed=seed)
    if len(checkpoints) != len(data)*len(SEEDS)*len(MODES):
        raise ValueError("Incomplete comparison")
    p = OUT / "head_freeze.json"
    run.once(p, run.stable(dict(registration=key, heads=checkpoints)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(p), summary=summary(results),
        by_site=results, counts_matched_per_query=True, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def heartbeat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] != "fit":
                print(row, flush=True)

        reg = register()
        if args.replay:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(dir=PRIVATE) as t:
                n = execute(*reg, heartbeat, True, Path(t))
        else:
            n = execute(*reg, heartbeat, False, PRIVATE)
        heartbeat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
