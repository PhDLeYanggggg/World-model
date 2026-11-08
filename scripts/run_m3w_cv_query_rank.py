"""Re-use nested forecasters to isolate CV reference and query-ranking loss."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_nested_velocity_selection as parent
from src.world_model import m3w_cv_query_rank as model

np, torch, run, prev, base = parent.np, parent.torch, parent.run, parent.prev, parent.base
data_source = parent.parent.parent.parent
forecast = parent.parent.parent.model
PUBLIC = parent.PUBLIC.parent/"native_cv_query_rank_v1"
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/"registration.json"
FILES = (Path(__file__), Path(model.__file__), ROOT/"tests/test_m3w_cv_query_rank.py", PUBLIC/"protocol.md")
CONTRASTS = ("cv_row", "ols4_row", "ols4_rank", "min_cv_change", "prefix", "prior_ols4", "random")
STEPS = 500


def register():
    cut, jobs, outer = parent.register()
    nested = prev.receipt(parent.PUBLIC, "train")
    prev.receipt(parent.PUBLIC, "replay")
    if nested["registration"] != run.sha(parent.REG) or not nested["outer_excluded"]:
        raise ValueError("Require checked nested lineage")
    paths = (*FILES, parent.REG, parent.PUBLIC/"train_execution.json", parent.PUBLIC/"replay_execution.json")
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in paths], arms=model.ARMS,
        steps=STEPS, seeds=prev.SEEDS, primary="cv_rank_vs_no_edit_cv", direct=CONTRASTS,
        pairs_per_query=16, bootstrap=3000, fallback="cv", candidate_frozen=True,
        independent_confirmation=False, original_rule_unchanged=True, external_access=False)))
    return cut, jobs, outer, nested


def load_data(jobs, beat):
    d = data_source.data(jobs, beat)
    queries = data_source.old.fitting_layout(jobs[0]["identity"], d)
    ego, neighbors, masks, basis, scale, ref = forecast.inputs(d["geometry"])
    return d, queries, (ego, neighbors, masks["near"], basis, scale), ref


def banks(d, nested):
    with np.load(run.checked(nested["outer_bank"]), allow_pickle=False) as a, np.load(
            run.checked(nested["scores"]), allow_pickle=False) as b:
        for bank in (a, b):
            np.testing.assert_array_equal(bank["ids"], d["ids"])
        return a["delta"][:, parent.parent.model.ARMS.index("masked_plain")], b["scores"][:, parent.model.ARMS.index("gain_harm")]


def train(jobs, outer, nested, beat, resume):
    d, queries, past, ref = load_data(jobs, beat)
    sites = d["sites"].astype(str)
    diff = ref["ols4"]-ref["cv"]
    layout = parent.model.folds(sites)
    n = len(sites)
    x = tuple(map(torch.from_numpy, past))
    denom = np.empty(n)
    for q in queries:
        denom[q] = 12*past[-1][q].mean()
    bank, _ = banks(d, nested)
    lookup = {(i["identity"]["seed"], tuple(i["identity"]["excluded"])): i for i in nested["donors"]}
    outer_lookup = {(r["identity"]["seed"], r["identity"]["held"]): r for r in outer["fits"] if r["identity"]["arm"] == "masked_plain"}
    reg, fits = run.sha(REG), []
    scores = np.full((len(prev.SEEDS), len(model.ARMS), n), np.nan, np.float32)
    for k, seed in enumerate(prev.SEEDS):
        donor_predictions = {}
        for excluded in parent.model.donor_folds(sites):
            receipt = lookup[(seed, excluded)]
            ix = np.flatnonzero(np.isin(sites, excluded))
            state = torch.load(run.checked(receipt["checkpoint"]), map_location="cpu", weights_only=False)
            h = base.fit_source.source.array_hash(d["ids"][~np.isin(sites, excluded)])
            if state["identity"] != receipt["identity"] or state["identity"]["train_ids"] != h or state["step"] != parent.CFG["donor_steps"]:
                raise ValueError("Donor exclusion or checkpoint identity mismatch")
            net = forecast.VelocityModel().eval()
            net.load_state_dict(state["model"])
            with torch.inference_mode():
                p = np.concatenate([net(*(t[i] for t in x)).numpy()
                                    for i in np.array_split(ix, (len(ix)+1023)//1024)])
            donor_predictions[excluded] = {g: p[sites[ix] == g] for g in excluded}
        for held, source_map in layout.items():
            test = np.flatnonzero(sites == held)
            train = np.concatenate([np.flatnonzero(sites == g) for g in source_map])
            expected = base.fit_source.source.array_hash(d["ids"][sites != held])
            if outer_lookup[(seed, held)]["identity"]["train_ids"] != expected:
                raise ValueError("Outer forecast lineage does not match")
            delta = np.concatenate([donor_predictions[pair][g] for g, pair in source_map.items()])
            p = ref["ols4"][train]+delta
            feature = torch.from_numpy(model.feature_matrix(past, train, delta, diff))
            infer = torch.from_numpy(model.feature_matrix(past, test, bank[k, test], diff))
            targets = {floor: torch.from_numpy((forecast.scores(p, ref[floor][train],
                d["target"][train], d["valid"][train])["full_grid_lower"]/denom[train]).astype(np.float32))
                for floor in ("cv", "ols4")}
            local = np.full(n, -1, np.int64)
            local[train] = np.arange(len(train))
            qlocal = [local[q] for q in queries if sites[q[0]] != held]
            if any((q < 0).any() for q in qlocal):
                raise ValueError("Query crosses the held source")
            pair, group = model.pairs(qlocal, np.linalg.norm(delta, axis=-1).max(1)>1e-8,
                                     sites[train], seed)
            groups = [np.flatnonzero(group == g) for g in sorted(set(group))]
            pair_hash = base.fit_source.source.array_hash(d["ids"][train][pair])
            rng0 = None
            for j, arm in enumerate(model.ARMS):
                floor, kind = arm.split("_")
                ident = dict(held=held, seed=seed, arm=arm, registration=reg, inner_exclusions=source_map,
                             train_pairs=pair_hash, outer_train_ids=expected)
                name = PRIVATE/f"{seed}_{held}_{arm}.pt"
                torch.manual_seed(seed)
                net = model.network(feature.shape[1])

                def objective(net, ix):
                    at = pair[ix]
                    return model.pair_loss(net(feature[at]).squeeze(-1), targets[floor][at], kind == "rank")

                net, ck, rng = parent.train_network(net, objective, groups, name, ident, STEPS, beat, resume)
                if rng0 is not None:
                    base.fit_source.source.exact(rng0, rng)
                rng0 = rng
                with torch.inference_mode():
                    scores[k, j, test] = net(infer).flatten().numpy()
                fits.append(dict(identity=ident, checkpoint=ck, held_rows=len(test)))
    if len(fits) != 108 or not np.isfinite(scores).all():
        raise ValueError("Incomplete factorial score bank")
    path = PRIVATE/"freeze.json"
    receipt = parent.write_npz(PRIVATE/"scores.npz", ids=d["ids"], scores=scores)
    run.once(path, run.stable(dict(fits=fits, registration=reg, scores=receipt, past_only=True,
        source_exclusion=True, matched_training_pairs=True, cached_donors=len(lookup), cached_outer=len(outer_lookup))))
    return path


def score(jobs, cut, nested, freeze, beat):
    if freeze["registration"] != run.sha(REG) or not all(freeze[k] for k in
            ("past_only", "source_exclusion", "matched_training_pairs")):
        raise ValueError("Require frozen causal rankings")
    d, queries, _, ref = load_data(jobs, beat)
    bank, prior = banks(d, nested)
    with np.load(run.checked(freeze["scores"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(d["ids"], z["ids"])
        rankings = z["scores"]
    target, v = d["target"], d["valid"]
    y = np.where(v[..., None], target, 0.)
    _, src = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(src))
    for q in queries:
        w[q] = 1/len(q)
    mass = lambda x: np.bincount(src, weights=w*x, minlength=9)
    cv_error = np.where(v, np.linalg.norm(ref["cv"]-y, axis=2), 0.).sum(1)/v.sum(1).clip(1)
    denom = mass(cv_error)
    easy = (cv_error > 0.) & (cv_error <= cut["easy"])
    hard = v.any(1) & (cv_error >= cut["hard"])
    support = mass(easy*cv_error)
    prefix = model.prefix_control(d["geometry"][:, :16].reshape(-1, 8, 2))
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    records, paired = {}, {}
    for k, seed in enumerate(prev.SEEDS):
        p = ref["ols4"]+bank[k]
        gain = forecast.scores(p, ref["cv"], target, v)
        legal = np.linalg.norm(bank[k], axis=-1).max(1)>1e-8
        ranks = dict(zip(model.ARMS, rankings[k], strict=True))
        ranks.update(oracle=gain["full_grid_lower"], prior_ols4=prior[k], prefix=prefix,
            min_cv_change=-np.linalg.norm(p-ref["cv"], axis=2).mean(1),
            random=np.random.default_rng(seed+4533).random(len(src)))
        chosen = {a: parent.model.choose(s, legal, queries) for a, s in ranks.items()}
        count = [chosen["cv_rank"][q].sum() for q in queries]
        for a, c in chosen.items():
            np.testing.assert_array_equal([c[q].sum() for q in queries], count)
            total = {m: 100*mass(c*g)/denom for m, g in gain.items()}
            total["hard_contribution"] = 100*mass(c*hard*gain["full_grid_lower"])/denom
            local_harm = 100*mass(easy*c*np.maximum(-gain["observed"], 0.))[support>0]/support[support>0]
            records.setdefault(a, []).append(dict(**total, switches=int(c.sum()),
                positive=int((c & (gain["full_grid_lower"] > 0.)).sum()),
                worst_supported_known_easy_harm=float(local_harm.max()),
                undefined_easy=int((support <= 0.).sum())))
        a = np.where(chosen["cv_rank"][:, None, None], p, ref["cv"])
        for arm in CONTRASTS:
            b = np.where(chosen[arm][:, None, None], p, ref["cv"])
            result = forecast.scores(a, b, target, v)
            paired.setdefault(arm, []).append({m:100*mass(g)/denom for m, g in result.items()})
        beat(state="scored_frozen_seed", seed=seed)
    interval = lambda x: data_source.old.metric.intervals(np.stack(x, 1), draws)
    reduced = lambda rows: {k: interval([r[k] for r in rows]) if isinstance(rows[0][k], np.ndarray)
                            else [r[k] for r in rows] for k in rows[0]}
    return dict(arms={k: reduced(v) for k, v in records.items()},
        direct_cv_rank_vs={k: reduced(v) for k, v in paired.items()}, fits=len(freeze["fits"]),
        cached_donors=freeze["cached_donors"], cached_outer=freeze["cached_outer"], rows=len(src),
        queries=len(queries), seeds=prev.SEEDS, source_groups=len(denom), original_rule_unchanged=True,
        development_targets=False, external_access=False, independent_confirmation=False, deployment=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs, outer, nested = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit experiment before running")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def beat(**v):
            msg = dict(phase=a.phase, pid=os.getpid(), seconds=time.monotonic()-start, **v)
            base.fit_source.source.atomic(PRIVATE/"heartbeat.json", msg)
            if v.get("step", STEPS) == STEPS:
                print(msg, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError("Resource limit; atomic checkpoints preserved")

        mark = PUBLIC/f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            result = train(jobs, outer, nested, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            result = PUBLIC/"analysis.json"
            run.once(result, run.stable(score(jobs, cut, nested, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(result)))


if __name__ == "__main__":
    main()
