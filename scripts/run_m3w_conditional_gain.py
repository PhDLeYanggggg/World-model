"""Frozen-support conditional-gain study, fit sources only."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use the native arm64 .venv-pytorch interpreter.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_observability as prior
from src.world_model import m3w_conditional_gain as model
from src.world_model.m3w_cv_query_rank import pairs

run, prev, base, np, torch = prior.run, prior.prev, prior.base, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent / "native_conditional_gain_v1"
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(model.__file__), PUBLIC / "protocol.md",
         ROOT / "tests/test_m3w_conditional_gain.py")
STEPS = 500
TRAINER = prior.prior.prior.trainer
ORIGINAL = prior.prior.prior.prior
CONTRASTS = (("common_step_rank", "support_only"), ("common_step_rank", "conditional_point"),
             ("common_step_rank", "ols4"), ("conditional_point", "support_only"),
             ("common_step_rank", "minimum"), ("common_step_rank", "original"),
             ("original", "support_only"))


def register():
    cut, jobs = prior.register()
    old = prev.receipt(prior.PUBLIC, "train")
    prev.receipt(prior.PUBLIC, "replay")
    if old["registration"] != run.sha(prior.REG) or not (old["held_sources"] and old["past_only"]):
        raise ValueError("Prior support must be source-held")
    bound = (*FILES, prior.REG, prior.PUBLIC / "train_execution.json",
             prior.PUBLIC / "replay_execution.json", prior.PUBLIC / "support_swap_diagnostic.json",
             Path(TRAINER.__file__), Path(pairs.__code__.co_filename))
    run.once(REG, run.stable(dict(files=[run.artifact(f) for f in bound], steps=STEPS,
        settings=TRAINER.CFG, seeds=prev.SEEDS, arms=model.ARMS, pairs_per_query=16,
        contrasts=CONTRASTS, frozen_support=old["scores"], support_used_in_gain_fit=False,
        role="leave_one_fitting_source_out", bootstrap=3000, original_rule_unchanged=True,
        independent_confirmation=False)))
    return cut, jobs, old


def train(jobs, old, beat, resume):
    d, qs, pack = prior.prior.data(jobs, beat)
    x, _, scale, paths, delta, legal, _, _ = pack
    quality, rec = prior.quality_inputs(jobs[0]["identity"], d, qs, beat)
    if base.fit_source.source.array_hash(quality) != old["quality_hash"] or rec != old["quality_sources"]:
        raise ValueError("Rebuilt past quality differs from the frozen support pipeline")
    x = torch.from_numpy(np.column_stack((x, quality)))
    sites = sorted(set(d["sites"]))
    out = np.full((len(prev.SEEDS), len(model.ARMS), len(x), 12), np.nan, np.float32)
    static = np.full((len(x), 12), np.nan, np.float32)
    fits = []
    for held in sites:
        ir = np.flatnonzero(d["sites"] != held)
        iv = np.flatnonzero(d["sites"] == held)
        cut_q = [q for q in qs if d["sites"][q[0]] != held]
        local = np.full(len(x), -1, np.int64)
        local[ir] = np.arange(len(ir))
        qlocal = [local[q] for q in cut_q]
        if any((q < 0).any() for q in qlocal):
            raise ValueError("Held source reached the query training batch")
        debit, frac, _ = prior.model.components(paths["ols4"][ir], delta[ir],
            d["target"][ir], d["valid"][ir], scale[ir])
        valid = d["valid"][ir]
        mask = valid & legal[ir, None]
        weight = (debit*mask).sum(0)
        if (weight <= 0.).any():
            raise ValueError("No donor conditional gain support at a lead")
        static[iv] = (frac*debit*mask).sum(0)/weight
        debit, frac, observed = map(torch.from_numpy, (debit, frac, valid))
        train_x = x[ir]
        for k, seed in enumerate(prev.SEEDS):
            index, src = pairs(qlocal, legal[ir], d["sites"][ir], seed, per_query=16)
            groups = [np.flatnonzero(src == a) for a in sorted(set(src))]
            states = []
            for j, arm in enumerate(model.ARMS):
                ident = dict(seed=seed, arm=arm, held=str(held), registration=run.sha(REG),
                    train_ids=base.fit_source.source.array_hash(d["ids"][ir]),
                    train_pairs=base.fit_source.source.array_hash(d["ids"][ir][index]))
                torch.manual_seed(seed)
                net = model.network(x.shape[-1])

                def objective(net, ix):
                    a = index[ix]
                    return model.loss(net(train_x[a]), frac[a], debit[a], observed[a], arm)

                net, ck, rng = TRAINER.train_network(net, objective, groups,
                    PRIVATE / f"{held}_{seed}_{arm}.pt", ident, STEPS, beat, resume)
                states.append(rng)
                with torch.inference_mode():
                    out[k, j, iv] = net(x[iv]).numpy()
                fits.append(dict(identity=ident, checkpoint=ck, held_rows=len(iv),
                    donor_pairs=len(index),
                    pairs_with_common_support=int((valid[index[:, 0]] & valid[index[:, 1]]).any(1).sum())))
            base.fit_source.source.exact(*states)
    if len(fits) != 54 or not (np.isfinite(out).all() and np.isfinite(static).all()):
        raise ValueError("Incomplete conditional gain fits")
    path = PRIVATE / "freeze.json"
    bank = TRAINER.write_npz(PRIVATE / "mu.npz", ids=d["ids"], mu=out, donor_mean=static)
    run.once(path, run.stable(dict(fits=fits, registration=run.sha(REG), bank=bank,
        support=old["scores"], support_used_in_gain_fit=False, source_held=True, past_only=True)))
    return path


def score(jobs, cut, old, freeze, beat):
    if (freeze["registration"] != run.sha(REG) or freeze["support"] != old["scores"]
            or freeze["support_used_in_gain_fit"] or not (freeze["source_held"] and freeze["past_only"])):
        raise ValueError("Invalid fixed-support fit freeze")
    d, qs, pack = prior.prior.data(jobs, beat)
    _, _, scale, paths, delta, legal, magnitude, _ = pack
    floor, cv = paths["ols4"], paths["cv"]
    y, valid = d["target"], d["valid"]
    dist = np.linalg.norm(delta, axis=-1)
    _, fractional, _ = prior.model.components(floor, delta, y, valid, scale)
    known, endpoint = valid.any(1), valid[:, -1]
    cv_ade = prior.prior.prior.ade(cv, y, valid)[0]
    ols_ade = prior.prior.prior.ade(floor, y, valid)[0]
    easy = known & (cv_ade > 0.) & (cv_ade <= cut["easy"])
    hard = known & (cv_ade >= cut["hard"])
    sites, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(y))
    for q in qs:
        w[q] = 1. / len(q)

    def mass(values):
        return np.bincount(src, weights=w * values, minlength=len(sites))

    denom = mass(cv_ade)
    candidate = floor + delta
    realized = ORIGINAL.scores(candidate, floor, y, valid)["full_grid_lower"]
    with np.load(run.checked(freeze["bank"]), allow_pickle=False) as z, np.load(
            run.checked(old["scores"]), allow_pickle=False) as s:
        for v in (z, s):
            np.testing.assert_array_equal(d["ids"], v["ids"])
        mu, static = z["mu"], z["donor_mean"]
        i = prior.model.ARMS.index("factor_quality")
        p, original = s["support"][:, i], s["scores"][:, i]
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda data: ORIGINAL.parent.old.metric.intervals(np.stack(data, 1), draws)

    def summary(data):
        return {a: {m: interval([r[m] for r in rows]) for m in rows[0]} for a, rows in data.items()}

    modes = {key: dict(utility={}, contrasts={}, risk={}, changed_queries={})
             for key in ("support_matched", "count_matched")}
    error, support_cells = {}, []
    count = [min(2, int(legal[q].sum())) for q in qs]
    for k, seed in enumerate(prev.SEEDS):
        zero = -((1.-p[k])*dist).mean(1)
        r = dict(support_only=zero, minimum=-magnitude, oracle=realized, original=original[k],
                 donor_mean=model.combined(static, p[k], dist))
        r.update({a: model.combined(mu[k, i], p[k], dist) for i, a in enumerate(model.ARMS)})
        mask = valid & legal[:, None]
        for a, v in (("donor_mean", static), *zip(model.ARMS, mu[k], strict=True)):
            num = mass((np.abs(v-fractional)*dist*mask).sum(1))
            error.setdefault(a, []).append(num/mass((dist*mask).sum(1)))
        layout = model.support_layout(p[k].mean(1), zero, legal, qs)
        support_cells.append(dict(seed=seed, matched_bins=len(layout),
            flexible_bins=sum(0 < n < len(b) for b, n in layout)))
        for name, mode in modes.items():
            pick = ({a: model.matched_choices(v, layout) for a, v in r.items()} if name == "support_matched"
                else {a: prior.prior.model.choose(v, legal, qs) for a, v in r.items()})
            for a, at in pick.items():
                np.testing.assert_array_equal([at[q].sum() for q in qs], count)
                if name == "support_matched":
                    np.testing.assert_array_equal([at[b].sum() for b, _ in layout], [n for _, n in layout])
                mode["changed_queries"].setdefault(a, []).append(sum(
                    not np.array_equal(at[q], pick["support_only"][q]) for q in qs))
            pick["ols4"] = np.zeros(len(y), bool)
            preds = {a: np.where(at[:, None, None], candidate, floor) for a, at in pick.items()}
            for a, p_ in preds.items():
                g = ORIGINAL.scores(p_, floor, y, valid)
                ade, fde = prior.prior.prior.ade(p_, y, valid)
                vals = {n: 100.*mass(z)/denom for n, z in g.items()}
                vals.update(ade=mass(ade)/mass(known), fde=mass(fde*endpoint)/mass(endpoint),
                            hard_contribution=100.*mass(hard*g["full_grid_lower"])/denom)
                if a != "ols4":
                    vals["missing_step_fraction"] = mass(pick[a]*(~valid).mean(1))/mass(pick[a])
                    vals["predicted_missing_fraction"] = mass(pick[a]*(1.-p[k].mean(1)))/mass(pick[a])
                mode["utility"].setdefault(a, []).append(vals)
                mode["risk"].setdefault(a, []).append({n: [
                    float(100.*h/s) if s > 0. else None for h, s in zip(
                        mass(easy*np.maximum(ade-ref, 0.)), mass(easy*ref), strict=True)]
                    for n, ref in (("cv", cv_ade), ("ols4", ols_ade))})
            for a, b in CONTRASTS:
                result = ORIGINAL.scores(preds[a], preds[b], y, valid)
                mode["contrasts"].setdefault(f"{a}_vs_{b}", []).append(
                    {n: 100.*mass(z)/denom for n, z in result.items()})
        beat(state="seed_scored", seed=seed, switches=int(sum(count)))
    for mode in modes.values():
        u = mode["utility"]
        mode["regret"] = {a: interval([o["full_grid_lower"]-r["full_grid_lower"]
            for o, r in zip(u["oracle"], rows, strict=True)]) for a, rows in u.items()
            if a not in ("ols4", "oracle")}
        for key in ("utility", "contrasts"):
            mode[key] = summary(mode[key])
    return dict(modes=modes, conditional_mae={a: interval(v) for a, v in error.items()},
        support_matching=support_cells, rows=len(y), queries=len(qs), sources=len(sites),
        seeds=prev.SEEDS, switches_per_seed=int(sum(count)), fits=len(freeze["fits"]),
        frozen_support=True, support_used_in_gain_fit=False, original_rule_unchanged=True,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs, old = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit the registered experiment before running it")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def beat(**data):
            msg = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **data)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", msg)
            if data.get("step", STEPS) == STEPS:
                print(msg, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or msg["seconds"] > 43200:
                raise OSError("Resource limit; checkpoints preserved")

        receipt = PUBLIC / f"{a.phase}_execution.json"
        if receipt.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            out = train(jobs, old, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, old, prev.receipt(PUBLIC, "train"), beat)))
        run.once(receipt, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
