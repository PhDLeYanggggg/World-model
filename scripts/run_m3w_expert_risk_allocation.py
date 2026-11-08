"""Register, train and score causal expert-risk allocation on fitting sources."""

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
from scripts import run_m3w_replay_motion as prior
from scripts import run_m3w_nested_velocity_selection as trainer
from src.world_model import m3w_expert_risk_allocation as model

base, run, prev, np, torch = prior.base, prior.run, prior.prev, prior.np, prior.torch
PUBLIC = prior.PUBLIC.parent / "native_expert_risk_allocation_v1"
PRIVATE = prior.PRIVATE.parent / PUBLIC.name
REG = PUBLIC / "registration.json"
FILES = (Path(__file__), Path(model.__file__), PUBLIC / "protocol.md",
         ROOT / "tests/test_m3w_expert_risk_allocation.py")
STEPS = 500
PAIRS = (("independent", "ols4"), ("independent", "minimum"), ("independent", "backtest"),
         ("joint", "pair_additive"), ("joint", "independent"),
         ("independent", "risk_fit_gain"), ("risk_fit_gain", "gain_only"))


def register():
    cut, jobs = prior.register()
    prev.receipt(prior.PUBLIC, "replay")
    bound = (*FILES, prior.REG, prior.PUBLIC / "headroom.json",
             prior.PUBLIC / "replay_execution.json", Path(trainer.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(f) for f in bound], seeds=prev.SEEDS,
        steps=STEPS, training=trainer.CFG, fits=model.FITS, primary=PAIRS[:2] + PAIRS[3:4],
        pair_penalty=model.SCENE_PENALTY, shortlist=model.SHORTLIST, budget=2, bootstrap=3000,
        scope="leave_one_fitting_source_out", original_rule_unchanged=True, external_access=False)))
    return cut, jobs


def current_origins(identity, data):
    f = base.fit_source
    admission = f.prep.load_admission(ROOT, ROOT / f.read(f.source.CONFIG)["admission"])
    lookup = {c["clip"]: c for c in f.read(f.bank.old.HOME / "movingdronecrowd_intake_v1/analysis.json")["recordings"]}
    groups, origin, ids = [], [], []
    offset, cursor = 0, 0
    for node, v in sorted(admission["recordings"].items()):
        clip = lookup[v["clip"]]
        n = clip["window_support"]["obs8_pred12_native_stride1"]["history_complete_agent_instances"]
        start, offset = offset, offset + n
        if not n or v["exclusion_group"] not in identity["normalizers"]:
            continue
        path = ROOT / v["source_annotation_path"]
        f.prep.require_fit_record(admission, node, "fit", path)
        rec = f.prep.mot_record(path.read_bytes(), v["clip"], range(1, clip["image_frames"] + 1))
        stop = cursor + n
        for frame in range(7, rec.valid.shape[1] - 12):
            past = f.prep.past_inputs(rec, frame, f.prep.SPEC)
            h = past["positions"][past["history_mask"].all(1)]
            if len(h):
                at = np.arange(cursor, cursor + len(h))
                np.testing.assert_array_equal((h - h[:, -1:]).reshape(-1, 16).astype(np.float32),
                                              data["geometry"][at, :16])
                groups.append(at)
                origin.append(h[:, -1])
                cursor += len(h)
        if cursor != stop or not (data["sites"][stop-n:stop] == v["exclusion_group"]).all():
            raise ValueError("Fitting source coordinates did not align")
        ids.extend(range(start, start + n))
    np.testing.assert_array_equal(ids, data["ids"])
    return groups, np.concatenate(origin)


def ade(p, y, valid):
    error = np.linalg.norm(p - np.where(valid[..., None], y, 0.), axis=-1)
    return np.where(valid, error, 0.).sum(1) / valid.sum(1).clip(1), error[:, -1]


def train(jobs, beat, resume):
    d = prior.parent.data(jobs, beat)
    x, delta, scale, paths, legal, _ = model.design(d["geometry"])
    x, legal = torch.from_numpy(x), torch.from_numpy(legal)
    sites = sorted(set(d["sites"]))
    output = np.full((len(prev.SEEDS), len(model.FITS), *x.shape[:2], 3), np.nan, np.float32)
    fits = []
    for held in sites:
        it, iv = (np.flatnonzero(a) for a in (d["sites"] != held, d["sites"] == held))
        y, v = d["target"][it], d["valid"][it]
        cv, _ = ade(paths["cv"][it], y, v)
        cut = float(np.quantile(cv[cv > 0.], .25))
        target = torch.from_numpy(model.targets(paths["ols4"][it], delta[it], y, v, scale[it],
                                               (cv > 0.) & (cv <= cut)))
        groups = [np.flatnonzero(d["sites"][it] == s) for s in sites if s != held]
        for k, seed in enumerate(prev.SEEDS):
            states = []
            for j, arm in enumerate(model.FITS):
                ident = dict(seed=seed, arm=arm, held=str(held), registration=run.sha(REG),
                    train_ids=base.fit_source.source.array_hash(d["ids"][it]), donor_easy_cut=cut)
                torch.manual_seed(seed)
                net = model.network(x.shape[-1])
                net, ck, state = trainer.train_network(net,
                    lambda net, ix: model.loss(net, x[it[ix]], target[ix], legal[it[ix]], arm),
                    groups, PRIVATE / f"{seed}_{held}_{arm}.pt", ident, STEPS, beat, resume)
                states.append(state)
                with torch.inference_mode():
                    for at in np.array_split(iv, max(1, (len(iv) + 1023) // 1024)):
                        output[k, j, at] = model.predict(net, x[at]).numpy()
                fits.append(dict(identity=ident, checkpoint=ck, held_rows=len(iv)))
            base.fit_source.source.exact(*states)
    if len(fits) != 54 or not np.isfinite(output).all():
        raise ValueError("Incomplete held-source inference")
    artifact = trainer.write_npz(PRIVATE / "scores.npz", ids=d["ids"], scores=output)
    out = PRIVATE / "freeze.json"
    run.once(out, run.stable(dict(fits=fits, scores=artifact, registration=run.sha(REG),
        past_only=True, held_source=True, matched_draws=True)))
    return out


def pair_metrics(qs, selected, floor, pred, origins, y, valid):
    contraction, contraction_n = 0., 0
    known_gain, known_n = 0., 0
    for q in qs:
        a = q[selected[q] >= 0]
        for i in a:
            b = q[(q != i) & ((selected[q] < 0) | (q > i))]
            origin = origins[i] - origins[b]
            ref = floor[i] - floor[b]
            end = pred[i] - pred[b]
            contraction += float(np.maximum(np.linalg.norm(ref + origin[:, None], axis=-1)
                - np.linalg.norm(end + origin[:, None], axis=-1), 0.).sum())
            contraction_n += len(b) * floor.shape[1]
            truth = y[i] - y[b]
            known = valid[i] & valid[b]
            change = np.linalg.norm(ref - truth, axis=-1) - np.linalg.norm(end - truth, axis=-1)
            known_gain += float(np.where(known, change, 0.).sum())
            known_n += int(known.sum())
    return dict(contraction_sum=contraction, changed_pair_step_views=contraction_n,
                relative_error_gain_sum=known_gain, changed_known_pair_step_views=known_n)


def score(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not all(
            frozen[k] for k in ("past_only", "held_source", "matched_draws")):
        raise ValueError("Complete fit freeze required")
    d = prior.parent.data(jobs, beat)
    q, origin = current_origins(jobs[0]["identity"], d)
    _, delta, scale, paths, legal, replay = model.design(d["geometry"])
    y, v = d["target"], d["valid"]
    known, end = v.any(1), v[:, -1]
    cv, _ = ade(paths["cv"], y, v)
    easy, hard = (cv > 0.) & (cv <= cut["easy"]), known & (cv >= cut["hard"])
    floor = paths["ols4"]
    oracle_scores = model.targets(floor, delta, y, v, scale, easy)[..., 0] * (12. * scale[:, None])
    minimum = -np.linalg.norm(delta, axis=-1).mean(-1)
    with np.load(run.checked(frozen["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        bank = z["scores"].astype(np.float64)
    sites, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(src))
    for group in q:
        w[group] = 1. / len(group)

    def mass(x):
        return np.bincount(src, weights=w*x, minlength=len(sites))

    denominator = mass(cv)
    easy_mass = mass(easy * cv)
    weights = 12. * scale[:, None]
    switched = [min(2, int(legal[group].any(1).sum())) for group in q]
    changed, utility, direct, pair, easy_stats, absolute, choice = {}, {}, {}, {}, {}, {}, {}
    for k, seed in enumerate(prev.SEEDS):
        first, risk = bank[k]
        adjusted = (risk[..., 0] - .5 * (risk[..., 1] + risk[..., 2])) * weights
        select = model.choose(adjusted, legal, q, floor, delta, origin)
        for arm, rankings in (("gain_only", first[..., 0] * weights), ("risk_fit_gain", risk[..., 0] * weights),
                ("minimum", minimum), ("backtest", replay), ("oracle", oracle_scores)):
            select[arm] = model.choose(rankings, legal, q)["independent"]
        for s in select.values():
            np.testing.assert_array_equal([(s[g] >= 0).sum() for g in q], switched)
        select["ols4"] = np.full(len(src), -1)
        pred = {a: model.selected_paths(floor, delta, s) for a, s in select.items()}
        for name, p in pred.items():
            errs, ends = ade(p, y, v)
            g = prior.scores(p, floor, y, v)
            utility.setdefault(name, []).append({m: 100*mass(x)/denominator for m, x in g.items()})
            absolute.setdefault(name, []).append(dict(ade=mass(errs)/mass(known),
                fde=mass(end*ends)/mass(end),
                hard_contribution=100*mass(hard*g["full_grid_lower"])/denominator))
            extra = mass(easy * np.maximum(errs - cv, 0.))
            easy_stats.setdefault(name, []).append({str(site): float(100*extra[i]/easy_mass[i])
                if easy_mass[i] > 0. else None for i, site in enumerate(sites)})
            choice.setdefault(name, []).append(np.bincount(select[name]+1,
                minlength=len(model.WEIGHTS)+1).tolist())
            if name in ("independent", "pair_additive", "joint"):
                pair.setdefault(name, []).append(pair_metrics(q, select[name], floor, p, origin, y, v))
        for a, b in PAIRS:
            g = prior.scores(pred[a], pred[b], y, v)
            direct.setdefault(f"{a}_vs_{b}", []).append({m: 100*mass(x)/denominator for m, x in g.items()})
        for a in ("pair_additive", "joint"):
            changed.setdefault(a, []).append(sum(not np.array_equal(select[a][g],
                select["independent"][g]) for g in q))
        changed.setdefault("joint_vs_pair_additive", []).append(sum(not np.array_equal(select["joint"][g],
            select["pair_additive"][g]) for g in q))
        beat(state="scored", seed=seed, queries=len(q), changes=int(sum(switched)))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    interval = lambda v: prior.parent.old.metric.intervals(np.stack(v, 1), draws)

    def aggregate(d):
        return {a: {m: interval([r[m] for r in values]) for m in values[0]}
                for a, values in d.items()}

    oracle = [r["full_grid_lower"] for r in utility["oracle"]]
    regret = {a: interval([o - r["full_grid_lower"] for o, r in zip(oracle, values, strict=True)])
              for a, values in utility.items() if a not in ("oracle", "ols4")}
    return dict(utility=aggregate(utility), direct=aggregate(direct), absolute=aggregate(absolute),
        fixed_budget_regret=regret, easy_known_positive_harm=easy_stats, pair_diagnostics=pair,
        changed_queries_by_seed=changed, actions_by_seed=choice, rows=len(src), queries=len(q),
        switches_per_seed=int(sum(switched)), sources=len(sites), seeds=prev.SEEDS,
        fit_count=len(frozen["fits"]), matched_count_each_query=True, geometric_proxy_only=True,
        original_rule_unchanged=True, independent_confirmation=False, deployed=False,
        stage5c_executed=False, smc_enabled=False)


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
            raise ValueError("Commit registered sources before use")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def beat(**data):
            message = dict(pid=os.getpid(), phase=a.phase, seconds=time.monotonic()-start, **data)
            base.fit_source.source.atomic(PRIVATE / "heartbeat.json", message)
            if data.get("step", STEPS) == STEPS:
                print(message, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or message["seconds"] > 43200:
                raise OSError("Resource limit; completed checkpoints preserved")

        mark = PUBLIC / f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            result = train(jobs, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            result = PUBLIC / "analysis.json"
            run.once(result, run.stable(score(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(result)))


if __name__ == "__main__":
    main()
