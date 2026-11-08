"""Nested source exclusion for neural forecast selection."""

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
from scripts import run_m3w_velocity_objectives as parent
from src.world_model import m3w_nested_velocity_selection as model

np, torch, run, prev, base = parent.np, parent.torch, parent.run, parent.prev, parent.base
PUBLIC = parent.PUBLIC.parent/"native_nested_velocity_selection_v1"
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/"registration.json"
FILES = (Path(__file__), Path(model.__file__), ROOT/"tests/test_m3w_nested_velocity_selection.py", PUBLIC/"protocol.md")
CFG = dict(batch=128, donor_steps=800, selector_steps=500, lr=.001, checkpoint=100, threads=4)


def register():
    cut, jobs, _ = parent.register()
    frozen = prev.receipt(parent.PUBLIC, "train")
    prev.receipt(parent.PUBLIC, "replay")
    if frozen["registration"] != run.sha(parent.REG) or not frozen["past_only"]:
        raise ValueError("Expected verified outer forecasts")
    paths = (*FILES, parent.REG, parent.PUBLIC/"capacity_diagnostic.json", parent.PUBLIC/"train_execution.json")
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in paths], cfg=CFG,
        selector_arms=model.ARMS, donor_loss="masked_plain", seeds=prev.SEEDS, bootstrap=3000,
        role="nested_leave_source_out", outer_only_for_score=True, matched_count=1,
        no_threshold_search=True, original_rule_unchanged=True, external_access=False)))
    return cut, jobs, frozen


def train_network(net, objective, groups, path, identity, steps, beat, resume):
    rng = torch.Generator().manual_seed(identity["seed"]+7919)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=.0001)
    trace, step = [], 0
    if path.exists():
        if not resume:
            raise FileExistsError("Existing checkpoint requires --resume")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or not 0 < saved["step"] <= steps:
            raise ValueError("Invalid resume identity")
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        trace, step = saved["trace"], saved["step"]
    while step < steps:
        ix = base.native.draw_batch(groups, CFG["batch"], rng)
        value = objective(net, ix)
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite loss")
        opt.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        step += 1
        if step % CFG["checkpoint"] == 0:
            trace.append(float(value.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, trace=trace, model=net.state_dict(),
                            optimizer=opt.state_dict(), rng=rng.get_state()), tmp)
            tmp.replace(path)
            beat(state="checkpoint", step=step, loss=trace[-1], **identity)
    return net.eval(), run.artifact(path), rng.get_state()


def write_npz(path, **values):
    tmp = path.with_suffix(".pending.npz")
    np.savez(tmp, **values)
    tmp.replace(path)
    return run.artifact(path)


def training(jobs, frozen, beat, resume):
    d = parent.parent.parent.data(jobs, beat)
    sites = d["sites"].astype(str)
    ego, neighbors, masks, basis, scale, paths = parent.parent.model.inputs(d["geometry"])
    past = (ego, neighbors, masks["near"], basis, scale)
    x = tuple(map(torch.from_numpy, past))
    y, f, valid = map(torch.from_numpy, (d["target"], paths["ols4"].astype(np.float32), d["valid"]))
    with np.load(run.checked(frozen["predictions"]), allow_pickle=False) as bank:
        np.testing.assert_array_equal(d["ids"], bank["ids"])
        outer_bank = bank["delta"][:, parent.model.ARMS.index("masked_plain")]
    outer_fits = {(z["identity"]["held"], z["identity"]["seed"]): z for z in frozen["fits"] if z["identity"]["arm"] == "masked_plain"}
    layout = model.folds(sites)
    output = np.full((len(prev.SEEDS), len(model.ARMS), len(y)), np.nan, np.float32)
    donors, heads = [], []
    reg = run.sha(REG)
    for k, seed in enumerate(prev.SEEDS):
        forecasts = {}
        for a, b in model.donor_folds(sites):
            use = ~np.isin(sites, (a, b))
            train = np.flatnonzero(use)
            exclude = np.flatnonzero(~use)
            groups = [np.flatnonzero(sites[train] == g) for g in sorted(set(sites[train]))]
            ident = dict(seed=seed, kind="donor", excluded=[a, b], registration=reg,
                         train_ids=base.fit_source.source.array_hash(d["ids"][train]))
            name = PRIVATE/f"donor_{seed}_{a}_{b}.pt"
            torch.manual_seed(seed)
            net = parent.parent.model.VelocityModel()

            def objective(net, ix):
                at = train[ix]
                pred = f[at]+net(*(p[at] for p in x))
                return parent.model.objective(pred, f[at], y[at], valid[at], x[-1][at], "masked_plain")

            net, ck, _ = train_network(net, objective, groups, name, ident, CFG["donor_steps"], beat, resume)
            with torch.inference_mode():
                pred = np.concatenate([net(*(p[at] for p in x)).numpy()
                                       for at in np.array_split(exclude, (len(exclude)+1023)//1024)])
            forecasts[(a, b)] = {g: pred[sites[exclude] == g] for g in (a, b)}
            donors.append(dict(identity=ident, checkpoint=ck, predicted_rows=len(exclude)))
        for outer, mapping in layout.items():
            test = np.flatnonzero(sites == outer)
            train = np.concatenate([np.flatnonzero(sites == g) for g in mapping])
            # A selector-training row is predicted by a donor that has seen neither
            # its source nor the outer evaluation source.
            delta = np.concatenate([forecasts[pair][g] for g, pair in mapping.items()])
            group = [np.flatnonzero(sites[train] == g) for g in mapping]
            cached = outer_fits[(outer, seed)]
            expected = base.fit_source.source.array_hash(d["ids"][np.flatnonzero(sites != outer)])
            if cached["identity"]["train_ids"] != expected or cached["held_rows"] != len(test):
                raise ValueError("Outer forecast training roster does not match")
            features = torch.from_numpy(model.features(past, train, delta))
            labels = torch.from_numpy(model.targets(paths["ols4"][train], delta, d["target"][train],
                                                   d["valid"][train], scale[train]))
            infer = torch.from_numpy(model.features(past, test, outer_bank[k, test]))
            rngs = []
            for j, arm in enumerate(model.ARMS):
                ident = dict(seed=seed, kind="selector", arm=arm, held=outer, registration=reg,
                    train_ids=base.fit_source.source.array_hash(d["ids"][train]), inner_exclusions=mapping)
                name = PRIVATE/f"selector_{seed}_{outer}_{arm}.pt"
                torch.manual_seed(seed)
                net, ck, rng = train_network(model.network(features.shape[1]),
                    lambda net, ix: model.loss(net, features[ix], labels[ix], arm),
                    group, name, ident, CFG["selector_steps"], beat, resume)
                rngs.append(rng)
                with torch.inference_mode():
                    output[k, j, test] = model.selection_scores(net, infer, arm, x[-1][test]).numpy()
                heads.append(dict(identity=ident, checkpoint=ck, held_rows=len(test)))
            base.fit_source.source.exact(*rngs)
    if len(donors) != 108 or len(heads) != 54 or not np.isfinite(output).all():
        raise ValueError("Nested training or inference incomplete")
    score = write_npz(PRIVATE/"scores.npz", ids=d["ids"], scores=output)
    path = PRIVATE/"freeze.json"
    run.once(path, run.stable(dict(registration=reg, donors=donors, selectors=heads, scores=score,
        outer_bank=frozen["predictions"], outer_excluded=True, feature_past_only=True)))
    return path


def scoring(jobs, cut, frozen, beat):
    if frozen["registration"] != run.sha(REG) or not all(frozen[k] for k in ("outer_excluded", "feature_past_only")):
        raise ValueError("Expected nested held-source score bank")
    d = parent.parent.parent.data(jobs, beat)
    queries = parent.parent.parent.old.fitting_layout(jobs[0]["identity"], d)
    ref = parent.parent.model.inputs(d["geometry"])[-1]
    y, valid = d["target"], d["valid"]
    clean = np.where(valid[..., None], y, 0.)
    full = valid.all(1)
    _, site = np.unique(d["sites"], return_inverse=True)
    w = np.empty(len(y))
    for q in queries:
        w[q] = 1./len(q)
    mass = lambda val: np.bincount(site, weights=w*val, minlength=9)

    def ade(p):
        return np.where(valid, np.linalg.norm(p-clean, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)

    cv, ols = ade(ref["cv"]), ade(ref["ols4"])
    denom = mass(cv)
    easy = (cv > 0.) & (cv <= cut["easy"])
    hard = (cv >= cut["hard"]) & valid.any(1)
    with np.load(run.checked(frozen["scores"]), allow_pickle=False) as z, np.load(
            run.checked(frozen["outer_bank"]), allow_pickle=False) as b:
        for bank in (z, b):
            np.testing.assert_array_equal(bank["ids"], d["ids"])
        scores, bank = z["scores"], b["delta"][:, parent.model.ARMS.index("masked_plain")]
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    rows, contrast = {}, {}
    for k, seed in enumerate(prev.SEEDS):
        delta = bank[k]
        dist = np.linalg.norm(delta, axis=-1)
        legal = dist.max(1) > 1e-8
        candidates = ref["ols4"]+delta
        values = parent.parent.model.scores(candidates, ref["ols4"], y, valid)
        lower = values["full_grid_lower"]
        common = dict(zip(model.ARMS, scores[k], strict=True))
        common.update(random=np.random.default_rng(4533+seed).random(len(y)), min_change=-dist.mean(1), oracle=lower)
        chosen = {a: model.choose(v, legal, queries) for a, v in common.items()}
        expected = [chosen["gain"][q].sum() for q in queries]
        for c in chosen.values():
            np.testing.assert_array_equal([c[q].sum() for q in queries], expected)
        chosen["positive_score_only"] = model.choose(common["gain_harm"], legal, queries, optional=True)
        proposals = {a: np.where(c[:, None, None], candidates, ref["ols4"]) for a, c in chosen.items()}
        for a, c in chosen.items():
            easy_support = mass(easy*cv)
            harm = np.maximum(-values["observed"], 0.)
            easy_harm = 100*mass(easy*c*harm)[easy_support>0]/easy_support[easy_support>0]
            other = parent.parent.model.scores(proposals[a], ref["cv"], y, valid)
            rows.setdefault(a, []).append(dict(lower=100*mass(c*lower)/denom,
                vs_cv=100*mass(other["full_grid_lower"])/denom,
                observed=100*mass(c*values["observed"])/denom,
                hard_gain=100*mass(c*hard*lower)/denom,
                max_known_easy_harm=float(easy_harm.max()), undefined_easy=int((easy_support==0).sum()),
                switches=int(c.sum()), positive=int((c & (lower>0.)).sum()), fully_labeled=int((c & full).sum())))
        for a in ("gain", "random", "min_change"):
            q = parent.parent.model.scores(proposals["gain_harm"], proposals[a], y, valid)["full_grid_lower"]
            contrast.setdefault(a, []).append(100*mass(q)/denom)
        beat(state="scored_frozen_seed", seed=seed)
    ci = lambda v: parent.parent.parent.old.metric.intervals(np.stack(v, 1), draws)
    out = {}
    for a, r in rows.items():
        out[a] = {k: ci([t[k] for t in r]) if isinstance(r[0][k], np.ndarray) else [t[k] for t in r]
                  for k in r[0]}
    return dict(arms=out, direct_gain_harm_vs={k: ci(v) for k, v in contrast.items()},
        rows=len(y), queries=len(queries), source_groups=len(denom), seeds=prev.SEEDS,
        donor_fits=len(frozen["donors"]), selector_fits=len(frozen["selectors"]), cached_outer_fits=27,
        development_targets=False, external_access=False, independent_confirmation=False,
        original_rule_unchanged=True, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "train", "score", "replay"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cut, jobs, prior = register()
    if a.phase == "register":
        return
    for f in (*FILES, REG):
        if subprocess.check_output(["git", "show", f"HEAD:{f.relative_to(ROOT)}"], cwd=ROOT) != f.read_bytes():
            raise ValueError("Commit registration before executing")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG["threads"])
        torch.set_num_interop_threads(1)

        def beat(**v):
            e = dict(phase=a.phase, pid=os.getpid(), seconds=time.monotonic()-start, **v)
            base.fit_source.source.atomic(PRIVATE/"heartbeat.json", e)
            if "step" not in v or v["step"] in (CFG["donor_steps"], CFG["selector_steps"]):
                print(e, flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError("Resource guard stopped with checkpoints preserved")

        mark = PUBLIC/f"{a.phase}_execution.json"
        if mark.exists():
            prev.receipt(PUBLIC, a.phase)
            return
        if a.phase == "train":
            result = training(jobs, prior, beat, a.resume)
        else:
            if a.phase == "replay":
                prev.receipt(PUBLIC, "score")
            result = PUBLIC/"analysis.json"
            run.once(result, run.stable(scoring(jobs, cut, prev.receipt(PUBLIC, "train"), beat)))
        run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(result)))


if __name__ == "__main__":
    main()
