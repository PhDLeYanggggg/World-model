"""Compare query-local with recording-oracle supervision under a causal count cap."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_matched_bridge_swaps as prior
from src.world_model import m3w_recording_count as model

parent, run, np, torch = prior.parent, prior.run, prior.np, prior.torch
OUT = prior.OUT.parent / "recording_count_v2"
PRIVATE = prior.PRIVATE.parent / OUT.name
RANK_STEPS, HEAD_STEPS = 700, 500
RATE = .2
ARMS = ("archived_additive", "frozen_additive", "local", "recording", "local_free", "recording_free")


def register():
    key, a, h = prior.register()
    paths = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "exact_matched_capacity")]
    p, ph, c = [run.read(f) for f in paths]
    if (any(v["registration"] != key for v in (p, ph, c)) or p["heads"] != run.sha(paths[1])
            or p["confirmation_access"] or c["confirmation_access"] or not p["counts_matched_per_query"]):
        raise ValueError("Require frozen matched swap results")
    out = OUT / "registration.json"
    aborted = OUT.parent / "recording_count_v1" / "registration.json"
    run.once(out, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parent=[run.artifact(f) for f in paths], horizon=[8, 12], seeds=prior.SEEDS,
        previous_aborted=run.artifact(aborted), comparator_fix="archive decoder and shared nested-menu decoder are separate",
        rank_steps=RANK_STEPS, count_steps=HEAD_STEPS, rate=RATE, bank=2., workers=0, threads=4,
        fitting="inner producer excludes both donor and outer; count head uses only donor labels",
        teacher="integer two-reference maximin under original risk and aggregate count cap; omit unresolved",
        primary="recording versus local at same streaming rate cap; realized counts need not match",
        original_2pct=True, confirmation_access=False, no_held_tuning=True)))
    return run.sha(out), a, h, p, ph


def fit(identity, name, seed, net_builder, steps, batch, loss, replay, beat):
    file = PRIVATE / f"{name}.pt"
    torch.manual_seed(seed)
    net = net_builder()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+391)
    step, trace = 0, []
    if file.exists():
        z = torch.load(file, weights_only=True, map_location="cpu")
        if z["identity"] != identity or z["step"] not in range(100, steps+1, 100):
            raise ValueError("Recording count checkpoint changed")
        step, trace = z["step"], z["trace"]
        net.load_state_dict(z["net"])
        opt.load_state_dict(z["opt"])
        rng.bit_generator.state = z["rng"]
    if replay and step != steps:
        raise ValueError("Replay cannot train")
    for t in range(step+1, steps+1):
        l = loss(net, *batch(rng))
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % 100 == 0:
            trace.append(float(l.detach()))
            tmp = file.with_suffix(".pending")
            torch.save(dict(identity=identity, step=t, trace=trace, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(file)
            beat(phase="fit", name=name, step=t)
    return net.eval(), dict(identity=identity, artifact=run.artifact(file), trace=trace), steps-step


@torch.inference_mode()
def menus(d, bank, rank):
    _, _, _, x, _, legal = bank
    x0 = torch.from_numpy(x.reshape(-1, 1, x.shape[-1])).repeat(1, 2, 1)
    z = torch.zeros(len(x0))
    scores = rank(x0, z, z).numpy().reshape(legal.shape)
    rows, acts, feats, masks = [], [], [], []
    for ix, n in prior.groups(d):
        use = np.where(legal[ix], scores[ix], -np.inf)
        arm = use.argmax(1)
        order = np.argsort(-use[np.arange(n), arm], kind="stable")
        k = min(2, n-1, int(legal[ix].any(1).sum()))
        r = np.full(2, -1, dtype=int)
        a = r.copy()
        r[:k] = ix.start+order[:k]
        a[:k] = arm[order[:k]]
        top = np.zeros((2, x.shape[-1]), np.float32)
        top[:k] = x[r[:k], a[:k]]
        feats.append(np.r_[x[ix, 0].mean(0), top.ravel(), np.log1p(n), k == 1, k == 2])
        masks.append(np.arange(3) <= k)
        rows.append(r)
        acts.append(a)
    return dict(x=np.asarray(feats, np.float32), legal=np.asarray(masks), row=np.asarray(rows), arm=np.asarray(acts))


def labels(d, bank, menu, cuts):
    cv, ols, b, _, scale, _ = bank
    y, valid = d["y"], d["valid"]
    parts, offset = prior.model.label_parts(cv, ols, b, y, valid, cuts[0])
    raw = [prior.model.moments(cv, p, y, valid, cuts[0]) for p in b.swapaxes(0, 1)]
    signed = np.column_stack([m["easy_signed"] for m in raw])
    row, arm, legal = (menu[k] for k in ("row", "arm", "legal"))
    picked = np.where((row >= 0)[..., None], parts[row, arm], 0.)
    risk3 = np.where(row >= 0, signed[row, arm], 0.)
    mom = np.concatenate((np.zeros_like(picked[:, :1]), np.cumsum(picked, axis=1)), axis=1)
    risk = np.concatenate((mom[..., 2:4], np.c_[np.zeros(len(row)), np.cumsum(risk3, axis=1)][..., None]), axis=-1)
    lengths = np.array([12.*np.median(scale[ix]) for ix, _ in prior.groups(d)])
    base = np.array([offset[ix].sum() for ix, _ in prior.groups(d)])
    gain = mom[..., :2]
    # Match the earlier local proxy with an explicit zero-action counterfactual.
    value = np.minimum(gain[..., 0], gain[..., 1]+base[:, None])/lengths[:, None]
    value -= (2.*np.maximum(risk[..., 0], 0.)+4.*np.maximum(risk[..., 1], 0.))/lengths[:, None]
    value += .025*(mom[..., 4] > 0.)
    local = np.where(legal, value, -np.inf).argmax(1)
    recording = np.full(len(row), -1, dtype=int)
    receipt = Counter()
    for rec in np.unique(d["query_rec"]):
        on = d["query_rec"] == rec
        agents = d["rec"] == rec
        mass = np.linalg.norm(cv[agents]-y[agents], axis=-1)[valid[agents]].sum()
        budget = int(np.floor(RATE*legal[on, 1].sum()+1e-9))
        plan, kind = model.teacher(gain[on], risk[on], mom[on, :, 4] > 0., legal[on], base[on].sum(),
                                   .02*raw[0]["ref_easy"][agents].sum(), mass, budget)
        if kind == "verified_witness":
            a = np.full(len(cv), -1, dtype=int)
            # Scatter the label plan into the same registered menu for exact risk verification.
            index = np.arange(2) < plan[:, None]
            a[row[on][index]] = arm[on][index]
            paths = parent.selected(cv[agents], b[agents], a[agents])
            check = parent.score({k: d[k][agents] for k in ("y", "valid", "rec")}, cv[agents], ols[agents], paths, cuts)
            if check["dual_positive"] != 1:
                raise ValueError("Teacher witness fails original evaluator")
        if kind != "unresolved":
            recording[on] = plan
        receipt[kind] += 1
    return (local, recording), dict(receipt)


def batches(menu_data, which):
    tensors = []
    for m, y in menu_data:
        use = y[which] >= 0
        tensors.append(tuple(torch.from_numpy(v[use]) for v in (m["x"], m["legal"], y[which])))
    tensors = [v for v in tensors if len(v[0])]

    def batch(rng):
        ids = rng.integers(len(tensors), size=128)
        pick = [rng.integers(len(v[0]), size=(ids == i).sum()) for i, v in enumerate(tensors)]
        return tuple(torch.cat([t[j][i] for t, i in zip(tensors, pick, strict=True)]) for j in range(3))
    return batch


def load(meta, builder, steps):
    z = torch.load(run.checked(meta["artifact"]), map_location="cpu", weights_only=True)
    if z["identity"] != meta["identity"] or z["step"] != steps:
        raise ValueError("Upstream checkpoint identity changed")
    net = builder()
    net.load_state_dict(z["net"])
    return net.eval()


@torch.inference_mode()
def evaluate(d, bank, menu, hurdle, criterion, rank, nets, cuts):
    cv, ols, b, x, _, legal = bank
    q = d["query"]
    a = parent.model.bank_decode(parent.predict(hurdle, bank), legal, q, [criterion])[0]
    counts = dict(frozen_additive=np.bincount(q[a >= 0], minlength=len(menu["x"])))
    for name, net in nets.items():
        logits = net(torch.from_numpy(menu["x"]), torch.from_numpy(menu["legal"])).numpy()
        counts[name+"_free"] = logits.argmax(1)
        counts[name] = model.streaming_count(logits, menu["legal"], d["query_rec"], d["starts"], RATE)
    archive = prior.decode(d, x, legal, a, {"archive": rank})["archive"]
    archive_path = parent.selected(cv, b, archive)
    paths = {"archived_additive": archive_path}
    scores = {"archived_additive": parent.score(d, cv, ols, archive_path, cuts)}
    np.testing.assert_array_equal(np.bincount(q[archive >= 0], minlength=len(menu["x"])), counts["frozen_additive"])
    for name, c in counts.items():
        use = np.arange(2) < c[:, None]
        at = np.full(len(cv), -1, dtype=int)
        at[menu["row"][use]] = menu["arm"][use]
        np.testing.assert_array_equal(np.bincount(q[at >= 0], minlength=len(c)), c)
        paths[name] = parent.selected(cv, b, at)
        scores[name] = parent.score(d, cv, ols, paths[name], cuts)
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    gain = lambda a, b: float(100.*parent.transfer.paired(paths[b], paths[a], y, valid)[1].sum()/mass)
    return dict(scores=scores, count_hist={k: np.bincount(v, minlength=3).tolist() for k, v in counts.items()},
                nested_archive_path_differences=int((paths["frozen_additive"] != archive_path).any((1, 2)).sum()),
                direct=dict(recording_vs_local=gain("recording", "local"),
                            recording_free_vs_local_free=gain("recording_free", "local_free"),
                            recording_vs_frozen=gain("recording", "frozen_additive")))


def execute(key, a, h, p, ph, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        source = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(source, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    rank_builder = lambda: prior.model.SwapValue(False)
    result, heads, rank_heads, steps = {}, {}, {}, 0
    for held in sites:
        donors = [s for s in sites if s != held]
        cut = lambda fit: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in fit]), [.25, .75])
        outer_cut = cut(donors)
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            train, teacher_audit = [], {}
            for donor in donors:
                omit = sorted((held, donor))
                fit_sites = [s for s in sites if s not in omit]
                cuts = cut(fit_sites)
                name = "rank_"+"_".join(omit)+f"_{seed}"
                ident = dict(registration=key, omitted=omit, fit=fit_sites, seed=seed)
                bank_train = []

                def rb(rng):
                    if not bank_train:
                        bank_train.extend(prior.prepare(sites[s], banks[s], cuts[0]) for s in fit_sites)
                    return prior.batch(bank_train, rng)

                loss = lambda net, x, y, pair, n: prior.model.objective(net(x, pair, n), y, True)
                rank, rank_heads[name], n = fit(ident, name, seed, rank_builder, RANK_STEPS, rb, loss, replay, beat)
                steps += n
                menu = menus(sites[donor], banks[donor], rank)
                y, teacher_audit[donor] = labels(sites[donor], banks[donor], menu, cuts)
                train.append((menu, y))
                beat(phase="donor", held=held, seed=seed, donor=donor, teacher=teacher_audit[donor])
            width = train[0][0]["x"].shape[-1]
            nets = {}
            for i, mode in enumerate(("local", "recording")):
                name = f"count_{held}_{seed}_{mode}"
                ident = dict(registration=key, held=held, fit=donors, seed=seed, mode=mode)
                nets[mode], heads[name], n = fit(ident, name, seed, lambda: model.CountHead(width), HEAD_STEPS,
                    batches(train, i), lambda net, x, legal, y: model.objective(net(x, legal), y), replay, beat)
                steps += n
            rank = load(ph["heads"][f"{held}/{seed}/additive_rank"], rank_builder, prior.STEPS)
            hurdle = load(h["heads"][f"{held}/{seed}/full"], parent.model.Hurdle, parent.STEPS)
            menu = menus(sites[held], banks[held], rank)
            out = evaluate(sites[held], banks[held], menu, hurdle, cell["criteria"][1], rank, nets, outer_cut)
            old = next(v for v in p["by_site"][held] if v["seed"] == seed)
            if run.stable(out["scores"]["archived_additive"]) != run.stable(old["scores"]["additive_rank"]):
                raise ValueError("Frozen ranker or counts changed")
            result[held].append(dict(seed=seed, teacher=teacher_audit, **out))
            beat(phase="held", held=held, seed=seed)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    rows = [r for cells in result.values() for r in cells]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    summary = dict(arms={k: dict(vs_cv=ci(lambda r: r["scores"][k]["vs_cv"]),
        vs_ols4=ci(lambda r: r["scores"][k]["vs_ols4"]),
        **{f: sum(r["scores"][k][f] for r in rows) for f in ("edits", "dual_positive", "risk_pass", "views", "undefined")}) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in rows[0]["direct"]},
        nested_archive_path_differences=sum(r["nested_archive_path_differences"] for r in rows))
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, count_heads=heads, rank_heads=rank_heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=summary,
        by_site=result, original_2pct=True, confirmation_access=False)))
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

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] not in ("fit", "donor"):
                print(row, flush=True)

        n = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
