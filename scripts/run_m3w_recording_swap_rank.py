"""Train cross-query gain preferences on swaps with identical recording credit."""

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
from scripts import run_m3w_donor_risk_readout as prior
from src.world_model import m3w_recording_swap_rank as model

base, parent, run, torch, study = prior.base, prior.parent, prior.run, prior.torch, prior.study
OUT = prior.OUT.parent / "recording_swap_rank_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
MODES = ("point", "swap_rank")
ARMS = ("prefix", "gain_only", "factorized", *MODES)
STEPS = 800


def register():
    key, a, ph, p, h, f, n, up = prior.register()
    paths = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    old, heads = [run.read(x) for x in paths]
    if (any(x["registration"] != key for x in (old, heads)) or old["heads"] != run.sha(paths[1])
            or old["confirmation_access"] or not old["original_2pct"]):
        raise ValueError("Require bound pair-excluded donors and online reference")
    path = OUT / "registration.json"
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in paths], modes=MODES, seeds=study.prior.SEEDS, steps=STEPS, native_steps=[8, 12],
        training="same legal actions and point loss; rank adds causal-anchor cross-query swaps with label-aware margins",
        matching="exact per-recording realized totals of frozen gain-only online policy",
        scoring="frozen factorized risk terms and two-reference gain score; unchanged credit and recording evaluator",
        primary="online and matched comparison of swap rank versus point-only with identical risk",
        no_threshold_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(path), a, ph, p, h, f, n, up, heads


def build(d, bank, menu, x, scale, cut, reference, proxy, seed):
    legal, rec, t = menu["legal"], d["query_rec"], d["starts"]
    anchor = [study.model.streaming_count(s, legal, rec, t) for s in (reference, proxy)]
    edges = model.swaps(anchor, legal, rec, np.random.default_rng(seed+511))
    if len(edges):
        np.testing.assert_array_equal(rec[edges[:, 0]], rec[edges[:, 1]])
    parts = model.components(bank[0], bank[2], d["y"], d["valid"], menu["row"], menu["arm"])
    margin = model.target(parts, edges, scale)
    y = base.labels(d, bank, menu, cut, scale)[..., :2]
    row = np.flatnonzero(legal[:, 1])
    return dict(x=x, y=y, legal=legal, rows=row, scale=scale, edges=edges, margin=margin, parts=parts)


def load_pair(meta, builder):
    z = torch.load(run.checked(meta["artifact"]), map_location="cpu", weights_only=True)
    if run.stable(z["identity"]) != meta["identity"] or z["step"] != prior.STEPS:
        raise ValueError("Pair checkpoint identity changed")
    net = builder()
    net.load_state_dict(z["net"])
    return net.eval()


def batches(sites):
    points = [v for v in sites if len(v["rows"])]
    pairs = [v for v in sites if len(v["edges"])]

    def batch(rng):
        gids = rng.integers(len(points), size=128)
        group = [(v, rng.choice(v["rows"], size=int((gids == i).sum()))) for i, v in enumerate(points)]
        xyz = [np.concatenate([v[k][at] for v, at in group]) for k in ("x", "y", "legal")]
        left, right, count, weights, m = [], [], [], [], []
        if pairs:
            ix = rng.integers(len(pairs), size=96)
            for i, p in enumerate(pairs):
                sample = rng.integers(len(p["edges"]), size=int((ix == i).sum()))
                a, b, c = p["edges"][sample].T
                left.append(p["x"][a])
                right.append(p["x"][b])
                count.append(c)
                ab = np.column_stack((p["scale"][a], p["scale"][b]))
                weights.append((ab/ab.mean(1, keepdims=True)).astype(np.float32))
                m.append(p["margin"][sample])
        else:
            left = right = [xyz[0][:0]]
            count = [np.empty(0, dtype=np.int64)]
            weights = [np.empty((0, 2), dtype=np.float32)]
            m = [np.empty(0, dtype=np.float32)]
        return tuple(torch.from_numpy(v) for v in (*xyz, *(np.concatenate(v) for v in (left, right, count, weights, m))))

    return batch


@torch.inference_mode()
def evaluate(d, b, menu, held_data, nets, signed, risk, proxy, cuts):
    cv, ols, bank = b[:3]
    rec, starts, legal = d["query_rec"], d["starts"], menu["legal"]
    r = signed.copy()
    r[..., [2, 3, 4, 5, 9]] = risk[..., [2, 3, 4, 5, 9]]
    score = base.model.score
    pred = {k: v(torch.from_numpy(held_data["x"])).numpy() for k, v in nets.items()}
    score0 = dict(prefix=proxy, gain_only=score(signed, risk=False), factorized=score(r))
    g = {"factorized": signed[..., :2], **pred}
    for k, v in pred.items():
        z = r.copy()
        z[..., :2] = v
        score0[k] = score(z)
    online = {k: study.model.streaming_count(s, legal, rec, starts) for k, s in score0.items()}
    match = {k: np.zeros(len(rec), dtype=int) for k in ARMS}
    oracle = np.zeros(len(rec), dtype=int)
    known = held_data["parts"][..., 0]
    lower = known-held_data["parts"][..., 1]
    for name in np.unique(rec):
        on = rec == name
        total = int(online["gain_only"][on].sum())
        if total:
            oracle[on] = base.exact_score_plan(lower[on], legal[on], total)
            for k, z in score0.items():
                v = base.exact_score_plan(z[on], legal[on], total)
                if not model.respects_tokens(v, legal[on, 1]):
                    raise ValueError("Illegal retrospective assignment")
                match[k][on] = v
    mass = np.linalg.norm(cv-d["y"], axis=-1)[d["valid"]].sum()

    def metrics(c):
        paths = {k: prior.prior.attribution.install(cv, bank, menu, v) for k, v in c.items()}
        contrasts = (("swap_rank", "point"), ("swap_rank", "factorized"), ("swap_rank", "prefix"),
                     ("point", "factorized"), ("point", "prefix"))
        return dict(scores={k: parent.score(d, cv, ols, p, cuts) for k, p in paths.items()},
            direct={f"{a}_vs_{f}": float(100.*parent.transfer.paired(paths[f], paths[a], d["y"], d["valid"])[1].sum()/mass)
                    for a, f in contrasts})

    i = np.arange(len(rec))
    ceiling = lower[i, oracle].sum()
    regret = {k: float(1200.*(ceiling-lower[i, c].sum())/mass) for k, c in match.items()}
    if min(regret.values()) < -1e-9:
        raise ValueError("Allocation cannot exceed the unconstrained gain oracle")
    a, z, c = held_data["edges"].T
    m = held_data["margin"]
    s = held_data["scale"]
    valid = m != 0.
    concordance = {}
    for k, v in g.items():
        delta = v[z, c].min(1)*s[z]-v[a, c].min(1)*s[a]
        concordance[k] = int((np.sign(delta[valid]) == np.sign(m[valid])).sum())
    use = legal.copy()
    use[:, 0] = False
    err = {k: float(np.abs(v[..., 0][use]-held_data["y"][..., 0][use]).mean()) for k, v in g.items()}
    return dict(online=metrics(online), matched=metrics(match), matched_regret_pp=regret,
        swap_pairs=len(m), certain_swaps=int(valid.sum()), swap_correct=concordance, action_gain_mae=err)


def execute(key, a, ph, p, h, f, n, up, heads, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d.update(query_rec=z["recording"], starts=z["start"])
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    cut = lambda ss: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in ss]), [.25, .75])
    rank = lambda: study.prior.model.SwapValue(False)
    width = 3*study.prior.model.WIDTH+3+2+base.model.PEER_WIDTH
    factor = lambda: prior.prior.model.Factorized(width)
    rows, frozen, steps = {}, {}, 0
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            train = []
            for s in donors:
                omitted = sorted((held, s))
                nm = "_".join(omitted)+f"_{seed}"
                r, m = up["rank_heads"]["rank_"+nm], heads["pair_moments"]["moments_"+nm]
                if any(x["identity"]["omitted"] != omitted for x in (r, m)):
                    raise ValueError("A donor predictor saw a held site")
                rk = study.load(r, rank, study.RANK_STEPS)
                net = load_pair(m, factor)
                dv, bv = sites[s], banks[s]
                menu = study.menus(dv, bv, rk)
                x, scale, proxy = base.features(dv, bv, menu)
                with torch.inference_mode():
                    ref = net(torch.from_numpy(x["motion"])).numpy()[..., :2].min(-1)
                train.append(build(dv, bv, menu, x["motion"], scale, cut([k for k in donors if k != s])[0], ref, proxy, seed))
            if not any((v["margin"] != 0.).any() for v in train):
                raise ValueError("No learnable cross-query preferences")
            batch = batches(train)
            nets = {}
            for mode in MODES:
                nm = f"{held}_{seed}_{mode}"
                ident = dict(registration=key, held=held, donors=donors, seed=seed, mode=mode)
                nets[mode], frozen[nm], u = study.fit(ident, nm, seed, lambda: model.RankValue(width), STEPS, batch,
                    lambda net, *x: model.objective(net, *x, rank=mode == "swap_rank"), replay, beat)
                steps += u
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], rank, study.prior.STEPS)
            bank = banks[held]
            menu = study.menus(d, bank, rk)
            x, scale, proxy = base.features(d, bank, menu)
            motion = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(width), base.STEPS)
            risk = study.load(n["heads"][f"{held}_{seed}"], factor, base.STEPS)
            with torch.inference_mode():
                t = torch.from_numpy(x["motion"])
                signed, risk_value = motion(t).numpy(), prior.prior.model.moments_for_score(risk(t))
            cuts = cut(donors)
            hh = build(d, bank, menu, x["motion"], scale, cuts[0], base.model.score(signed, risk=False), proxy, seed)
            value = evaluate(d, bank, menu, hh, nets, signed, risk_value, proxy, cuts)
            old = next(v for v in f["by_site"][held] if v["seed"] == seed)
            for k in ("online", "matched"):
                if run.stable(value[k]["scores"]["factorized"]) != run.stable(old[k]["scores"]["factorized_risk"]):
                    raise ValueError("Frozen factorized control changed")
            ref = next(v for v in p["by_site"][held] if v["seed"] == seed)
            if run.stable(value["online"]["scores"]["prefix"]) != run.stable(ref["online"]["past_proxy"]):
                raise ValueError("Frozen prefix control changed")
            rows[held].append(dict(seed=seed, training_swaps=sum(len(v["margin"]) for v in train),
                training_certain_swaps=sum(int((v["margin"] != 0.).sum()) for v in train), **value))
            beat(phase="held", held=held, seed=seed)
    return finish(key, rows, frozen, steps)


def finish(key, rows, frozen, steps):
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))
    flat = [r for v in rows.values() for r in v]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    out = {s: dict(arms={k: {f: sum(v[s]["scores"][k][f] for v in flat) for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")}
        for k in ARMS}, direct={k: ci(lambda v: v[s]["direct"][k]) for k in flat[0][s]["direct"]}) for s in ("online", "matched")}
    out["cv_gain_only_oracle_regret_pp"] = {k: ci(lambda r: r["matched_regret_pp"][k]) for k in ARMS}
    out["swap_correct"] = {k: sum(r["swap_correct"][k] for r in flat) for k in (*MODES, "factorized")}
    out["gain_mae"] = {k: ci(lambda v: v["action_gain_mae"][k]) for k in (*MODES, "factorized")}
    for k in ("swap_pairs", "certain_swaps", "training_swaps", "training_certain_swaps"):
        out[k] = sum(v[k] for v in flat)
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=out, by_site=rows,
        original_2pct=True, confirmation_access=False, matching_label_blind_not_online=True)))
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

        n = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=n, replay=args.replay)


if __name__ == "__main__":
    main()
