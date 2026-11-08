"""Fit, freeze and replay fixed-candidate recipient ranking."""

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
from scripts import run_m3w_expert_risk_allocation as prior
from src.world_model import m3w_recipient_top2 as model

run, prev, base, np, torch = prior.run, prior.prev, prior.base, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent / "native_recipient_top2_v1"
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(model.__file__), ROOT / "tests/test_m3w_recipient_top2.py", PUBLIC / "protocol.md")
CFG = dict(batch_queries=16, updates=600, checkpoint=100, lr=.001, torch_threads=4, workers=0)
CONTRASTS = (("top2", "ols4"), ("top2", "minimum"), ("top2", "point"),
             ("top2_context", "top2"), ("top2_context", "ols4"), ("top2_context", "minimum"),
             ("top2", "prefix"), ("top2", "uniform"))


def register():
    cut, jobs = prior.register()
    prev.receipt(prior.PUBLIC, "replay")
    bound = (*FILES, prior.REG, prior.PUBLIC / "regret_diagnostic.json",
             prior.PUBLIC / "replay_execution.json", Path(prior.model.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in bound], cfg=CFG, arms=model.ARMS,
        seed=prev.SEEDS, bootstrap=3000, amplitude=model.AMPLITUDE, matches=CONTRASTS,
        role="leave_one_fitting_source_out", independent_confirmation=False, original_rule_unchanged=True)))
    return cut, jobs


def data(jobs, beat):
    d = prior.prior.parent.data(jobs, beat)
    q = prior.prior.parent.old.fitting_layout(jobs[0]["identity"], d)
    return d, q, model.inputs(d["geometry"], q)


def pack(q, eligible, sites):
    ids = np.zeros((len(q), max(map(len, q))), np.int64)
    mask = np.zeros_like(ids, bool)
    for i, g in enumerate(q):
        if len(set(sites[g])) != 1:
            raise ValueError("A query cannot cross sources")
        ids[i, :len(g)] = g
        mask[i, :len(g)] = eligible[g]
    return ids, mask, sites[ids[:, 0]]


def fit(net, x, y, mask, groups, ident, beat, resume):
    optimizer = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=.0001)
    rng = torch.Generator().manual_seed(ident["seed"] + 7919)
    path = PRIVATE / f'{ident["held"]}_{ident["seed"]}_{ident["arm"]}.pt'
    step, trace = 0, []
    if path.exists():
        if not resume:
            raise FileExistsError("Use --resume for existing checkpoints")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != ident or not 0 < saved["step"] <= CFG["updates"]:
            raise ValueError("Checkpoint identity changed")
        net.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        step, trace = saved["step"], saved["trace"]
    while step < CFG["updates"]:
        at = base.native.draw_batch(groups, CFG["batch_queries"], rng)
        value = model.train_loss(net, x[at], y[at], mask[at], ident["arm"])
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % CFG["checkpoint"] == 0:
            trace.append(float(value.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, model=net.state_dict(), optimizer=optimizer.state_dict(),
                            rng=rng.get_state(), step=step, trace=trace), tmp)
            tmp.replace(path)
            beat(state="checkpoint", step=step, held=ident["held"], arm=ident["arm"],
                 seed=ident["seed"], loss=trace[-1])
    return net.eval(), run.artifact(path), rng.get_state()


def train(jobs, beat, resume):
    d, q, packed = data(jobs, beat)
    x, context, scale, paths, change, legal, _, _ = packed
    ids, mask, qsite = pack(q, legal, d["sites"])
    sites = sorted(set(qsite))
    scores = np.full((len(prev.SEEDS), len(model.ARMS), len(x)), np.nan, np.float32)
    xs = dict(point=x, top2=x, top2_context=context)
    fits = []
    for held in sites:
        keep = qsite != held
        iq, ir = ids[keep], np.flatnonzero(d["sites"] != held)
        iv = np.flatnonzero(d["sites"] == held)
        loss_row = prior.prior.scores(paths["ols4"][ir] + change[ir], paths["ols4"][ir],
            d["target"][ir], d["valid"][ir])["full_grid_lower"] / scale[ir]
        local = np.full(len(x), -1, np.int64)
        local[ir] = np.arange(len(ir))
        mask_train = mask[keep]
        if (local[iq[mask_train]] < 0).any():
            raise ValueError("A training query includes a held row")
        y = torch.from_numpy(np.where(mask_train, loss_row[local[iq]], 0.).astype(np.float32))
        m = torch.from_numpy(mask_train)
        groups = [np.flatnonzero(qsite[keep] == s) for s in sites if s != held]
        training_x = {key: torch.from_numpy(xs[key][iq]) for key in ("point", "top2_context")}
        for k, seed in enumerate(prev.SEEDS):
            states = []
            for a, arm in enumerate(model.ARMS):
                ident = dict(seed=seed, arm=arm, held=str(held), registration=run.sha(REG),
                    train_rows_sha256=base.fit_source.source.array_hash(d["ids"][ir]))
                torch.manual_seed(seed)
                net = model.network(x.shape[-1])
                net, ck, rng = fit(net, training_x["top2_context" if arm == "top2_context" else "point"],
                                  y, m, groups, ident, beat, resume)
                states.append(rng)
                with torch.inference_mode():
                    scores[k, a, iv] = net(torch.from_numpy(xs[arm][iv])).flatten().numpy()
                fits.append(dict(identity=ident, checkpoint=ck, held_rows=len(iv)))
            for state in states[1:]:
                base.fit_source.source.exact(states[0], state)
    if len(fits) != 81 or not np.isfinite(scores).all():
        raise ValueError("Incomplete held-source recipient scores")
    artifact = prior.trainer.write_npz(PRIVATE / "scores.npz", ids=d["ids"], scores=scores)
    out = PRIVATE / "freeze.json"
    run.once(out, run.stable(dict(fits=fits, scores=artifact, registration=run.sha(REG),
        fixed_candidates=True, past_only=True, matched_query_draws=True, held_source=True)))
    return out


def score(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not all(
            frozen[k] for k in ("fixed_candidates", "past_only", "matched_query_draws", "held_source")):
        raise ValueError("Expected the frozen held-source scores")
    d, qs, packed = data(jobs, beat)
    _, _, _, paths, delta, legal, magnitude, prefix = packed
    y, valid, floor = d["target"], d["valid"], paths["ols4"]
    all_changed = floor + delta
    known, end = valid.any(1), valid[:, -1]
    cv_err, _ = prior.ade(paths["cv"], y, valid)
    ols_err, _ = prior.ade(floor, y, valid)
    easy = known & (cv_err > 0.) & (cv_err <= cut["easy"])
    hard = known & (cv_err >= cut["hard"])
    actual = prior.prior.scores(all_changed, floor, y, valid)["full_grid_lower"]
    sites, site = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(y))
    for q in qs:
        w[q] = 1. / len(q)

    def mass(v):
        return np.bincount(site, weights=w*v, minlength=len(sites))

    with np.load(run.checked(frozen["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], d["ids"])
        scores = z["scores"]
    denominator = mass(cv_err)
    if (denominator <= 0).any():
        raise ValueError("CV evaluation denominator missing")
    count = [min(2, int(legal[q].sum())) for q in qs]
    utility, direct, rates, risk, overlap = {}, {}, {}, {}, {}
    for k, seed in enumerate(prev.SEEDS):
        ranks = dict(zip(model.ARMS, scores[k], strict=True))
        ranks.update(minimum=-magnitude, prefix=prefix,
                     uniform=np.random.default_rng(seed+613).random(len(y)), oracle=actual)
        pick = {a: model.choose(value, legal, qs) for a, value in ranks.items()}
        for at in pick.values():
            np.testing.assert_array_equal([at[q].sum() for q in qs], count)
        pick["ols4"] = np.zeros(len(y), bool)
        pred = {a: np.where(at[:, None, None], all_changed, floor) for a, at in pick.items()}
        for a, p in pred.items():
            g = prior.prior.scores(p, floor, y, valid)
            ade, fde = prior.ade(p, y, valid)
            values = {k: 100. * mass(v) / denominator for k, v in g.items()}
            values.update(ade=mass(ade)/mass(known), fde=mass(fde*end)/mass(end),
                          hard_contribution=100.*mass(hard*g["full_grid_lower"])/denominator)
            utility.setdefault(a, []).append(values)
            rates.setdefault(a, []).append(int((pick[a] & (actual > 0.)).sum()))
            overlap.setdefault(a, []).append(int((pick[a] & pick["oracle"]).sum()))
            risk.setdefault(a, []).append({name: [
                float(100.*v/s) if s > 0 else None for v, s in zip(
                    mass(easy*np.maximum(ade-ref, 0.)), mass(easy*ref), strict=True)]
                for name, ref in (("cv", cv_err), ("ols4", ols_err))})
        for a, b in CONTRASTS:
            g = prior.prior.scores(pred[a], pred[b], y, valid)
            direct.setdefault(f"{a}_vs_{b}", []).append({m: 100.*mass(v)/denominator for m, v in g.items()})
        beat(state="score_seed", seed=seed, switches=sum(count))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda data: prior.prior.parent.old.metric.intervals(np.stack(data, 1), draws)

    def summarize(data):
        return {a: {m: interval([r[m] for r in rows]) for m in rows[0]} for a, rows in data.items()}

    regret = {a: interval([r["full_grid_lower"]-m["full_grid_lower"]
        for r, m in zip(utility["oracle"], rows, strict=True)]) for a, rows in utility.items()
        if a not in ("ols4", "oracle")}
    return dict(utility=summarize(utility), direct=summarize(direct), positive_actions=rates,
        overlap_with_oracle=overlap, same_nomination_regret=regret, easy_by_source_seed=risk,
        rows=len(y), query_count=len(qs), sources=sites.tolist(), seeds=prev.SEEDS,
        switches_per_seed=sum(count), fits=len(frozen["fits"]), matched_counts=True,
        no_role_changes=True, independent_confirmation=False, original_rule_unchanged=True,
        deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "train", "score", "replay"))
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    cut, jobs = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit registered code before training or scoring")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG["torch_threads"])
        torch.set_num_interop_threads(1)

        def beat(**values):
            msg = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **values)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", msg)
            if values.get("step", CFG["updates"]) == CFG["updates"]:
                print(msg, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or msg["seconds"] > 43200:
                raise OSError("Resource limit; checkpoints preserved")

        receipt = PUBLIC / f"{a.phase}_execution.json"
        if receipt.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            out = train(jobs, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            out = PUBLIC / "analysis.json"
            run.once(out, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(receipt, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))


if __name__ == "__main__":
    main()
