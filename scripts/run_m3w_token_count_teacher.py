"""Test whether matching teacher and inference budgets improves held decisions."""

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
from scripts import audit_m3w_streaming_menu_capacity as audit
from src.world_model import m3w_token_count_teacher as model

study, run, np, torch, parent = audit.study, audit.run, audit.np, audit.torch, audit.parent
OUT = audit.OUT.parent / "token_count_teacher_v1"
PRIVATE = audit.PRIVATE.parent / OUT.name
MODES = ("projected", "token_exact")
STEPS = 500


def register():
    ak, a, ph = audit.register()
    p = audit.OUT / "analysis.json"
    b = run.read(p)
    if b["registration"] != ak or b["confirmation_access"] or not b["original_2pct"]:
        raise ValueError("Require verified streaming ceiling")
    path = OUT / "registration.json"
    h = study.OUT / "head_freeze.json"
    heads = run.read(h)
    run.once(path, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        capacity=run.artifact(p), donor_rankers=run.artifact(h), modes=MODES, updates=STEPS,
        primary="token-exact teacher vs aggregate-quota teacher projected to token feasibility; both credit-conditioned",
        training_features="same causal query menu plus available credit", original_2pct=True,
        confirmation_access=False, no_held_tuning=True, seeds=study.prior.SEEDS, workers=0, threads=4)))
    return run.sha(path), a, ph, heads


def labels(d, bank, menu, cuts):
    cv, ols, b = bank[:3]
    y, valid = d["y"], d["valid"]
    prior = study.prior.model
    parts, offset = prior.label_parts(cv, ols, b, y, valid, cuts[0])
    raw = [prior.moments(cv, p, y, valid, cuts[0]) for p in b.swapaxes(0, 1)]
    signed = np.column_stack([m["easy_signed"] for m in raw])
    row, arm, legal = (menu[k] for k in ("row", "arm", "legal"))
    picked = np.where((row >= 0)[..., None], parts[row, arm], 0.)
    signed = np.where(row >= 0, signed[row, arm], 0.)
    mom = np.concatenate((np.zeros_like(picked[:, :1]), np.cumsum(picked, axis=1)), axis=1)
    risk = np.concatenate((mom[..., 2:4], np.c_[np.zeros(len(row)), np.cumsum(signed, axis=1)][..., None]), axis=-1)
    base = np.array([offset[ix].sum() for ix, _ in study.prior.groups(d)])
    rec = d["query_rec"]
    projected, token = (np.zeros(len(row), dtype=int) for _ in MODES)
    include = np.ones(len(row), dtype=bool)
    receipt = Counter()
    for r in np.unique(rec):
        on, agents = rec == r, d["rec"] == r
        mass = np.linalg.norm(cv[agents]-y[agents], axis=-1)[valid[agents]].sum()
        args = (mom[on, :, :2], risk[on], mom[on, :, 4] > 0., legal[on], base[on].sum(),
                .02*raw[0]["ref_easy"][agents].sum(), mass)
        old, status0 = study.model.teacher(*args, int(legal[on, 1].sum())//5)
        plan, status1 = model.teacher(*args)
        receipt[f"total_{status0}"] += 1
        receipt[f"token_{status1}"] += 1
        if "unresolved" in (status0, status1):
            include[on] = False
            continue
        projected[on], token[on] = old, plan
        if status1 == "verified_witness":
            choose = np.arange(2) < plan[:, None]
            at = np.full(len(cv), -1, dtype=int)
            at[row[on][choose]] = arm[on][choose]
            p = parent.selected(cv[agents], b[agents], at[agents])
            checked = parent.score({k: d[k][agents] for k in ("y", "valid", "rec")}, cv[agents], ols[agents], p, cuts)
            if checked["dual_positive"] != 1:
                raise ValueError("Token plan fails original risk evaluator")
    out = []
    for i, plan in enumerate((projected, token)):
        y, state = model.teacher_features(plan, legal, rec)
        if i == 1:
            np.testing.assert_array_equal(y, plan)
        else:
            receipt["projected_edits_removed"] = int((plan-y).sum())
        available = legal & (np.arange(3)[None]*5 <= state[:, None])
        out.append((dict(x=np.c_[menu["x"], state/10.].astype(np.float32), legal=available),
                    (np.where(include, y, -1),)))
    return out, dict(receipt)


@torch.inference_mode()
def evaluate(d, bank, menu, nets, cuts):
    cv, ols, b = bank[:3]
    n = len(menu["x"])
    rec = d["query_rec"]
    xs = np.c_[np.repeat(menu["x"], 11, axis=0), np.tile(np.arange(11)/10., n)].astype(np.float32)
    allowed = np.repeat(menu["legal"], 11, axis=0)
    paths, scores = {}, {}
    for k, net in nets.items():
        logit = net(torch.from_numpy(xs), torch.from_numpy(allowed)).numpy().reshape(n, 11, 3)
        count, _ = model.online(logit, menu["legal"], rec)
        use = np.arange(2) < count[:, None]
        at = np.full(len(cv), -1, dtype=int)
        at[menu["row"][use]] = menu["arm"][use]
        np.testing.assert_array_equal(np.bincount(d["query"][at >= 0], minlength=n), count)
        paths[k] = parent.selected(cv, b, at)
        scores[k] = parent.score(d, cv, ols, paths[k], cuts)
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    return dict(scores=scores, direct=float(100.*parent.transfer.paired(paths["projected"], paths["token_exact"], y, valid)[1].sum()/mass))


def execute(key, a, ph, heads, beat, replay):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    for s, d in sites.items():
        path = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(path, allow_pickle=False) as z:
            d["query_rec"] = z["recording"]
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    rank_builder = lambda: study.prior.model.SwapValue(False)
    results, models, steps = {}, {}, 0
    for held, d in sites.items():
        results[held] = []
        donors = [s for s in sites if s != held]
        cuts = lambda ss: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in ss]), [.25, .75])
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            train = {k: [] for k in MODES}
            counts = Counter()
            for donor in donors:
                omit = sorted((held, donor))
                name = "rank_"+"_".join(omit)+f"_{seed}"
                meta = heads["rank_heads"][name]
                if set(meta["identity"]["fit"]) != set(sites)-set(omit):
                    raise ValueError("Rank producer exposed to its held locations")
                rank = study.load(meta, rank_builder, study.RANK_STEPS)
                m = study.menus(sites[donor], banks[donor], rank)
                z, receipt = labels(sites[donor], banks[donor], m, cuts([s for s in sites if s not in omit]))
                counts.update(receipt)
                for k, v in zip(MODES, z, strict=True):
                    train[k].append(v)
                beat(phase="teacher", held=held, seed=seed, donor=donor)
            rank = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], rank_builder, study.prior.STEPS)
            menu = study.menus(d, banks[held], rank)
            nets = {}
            for mode in MODES:
                name = f"{held}_{seed}_{mode}"
                identity = dict(registration=key, held=held, donors=donors, seed=seed, mode=mode)
                nets[mode], models[name], n = study.fit(identity, name, seed,
                    lambda: study.model.CountHead(menu["x"].shape[-1]+1), STEPS, study.batches(train[mode], 0),
                    lambda net, x, legal, y: study.model.objective(net(x, legal), y), replay, beat)
                steps += n
            results[held].append(dict(seed=seed, teacher=counts, **evaluate(d, banks[held], menu, nets, cuts(donors))))
            beat(phase="held", held=held, seed=seed)
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    flat = [r for v in results.values() for r in v]

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in results.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    scores = {k: dict(vs_cv=ci(lambda r: r["scores"][k]["vs_cv"]), vs_ols4=ci(lambda r: r["scores"][k]["vs_ols4"]),
        **{f: sum(r["scores"][k][f] for r in flat) for f in ("edits", "dual_positive", "risk_pass", "undefined", "views")}) for k in MODES}
    t = Counter()
    for r in flat:
        t.update(r["teacher"])
    path = OUT / "head_freeze.json"
    run.once(path, run.stable(dict(registration=key, heads=models)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(path), summary=scores, teacher=t,
        direct_token_vs_projected=ci(lambda r: r["direct"]), by_site=results, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    # Reuse the checkpoint implementation, with this study's private directory.
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", row)
            if v["phase"] in ("held", "complete"):
                print(row, flush=True)

        steps = execute(*register(), beat, args.replay)
        beat(phase="complete", updates=steps, replay=args.replay)


if __name__ == "__main__":
    main()
