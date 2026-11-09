"""Learn the matched additive-relative pair objective on feasible donor slates."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_pair_ranking_targets as target
from src.world_model import m3w_slate_relative_gain as relative

rank, pair, np, torch, study, base = target.rank, target.pair, target.np, target.pair.torch, target.study, target.base
SITES, SEEDS = pair.SITES, pair.SEEDS
OUT, LOCAL = rank.OUT.parent/"opendd_conditional_pair_gain_v1", rank.LOCAL.parent/"opendd_conditional_pair_gain_v1"
STEPS, EVERY, BATCH, SHORTLIST, LABEL_LIMIT = 600, 50, 128, 8, 24
MODES = ("feasibility", "pair_gain")


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    old_key, cohorts, _ = rank.register()
    readout_path = rank.OUT/"summary.json"
    readout = study.read(readout_path)
    old = study.read(study.checked(readout["freeze"]))
    if (old["registration"] != old_key or readout["registration"] != old_key
            or old["held_outcomes_used"] or readout["reserved_outcomes_opened"]):
        raise ValueError("Require the verified matched ranking parent")
    p = OUT/"registration.json"
    base.immutable(p, dict(parent=study.artifact(readout_path), parent_freeze=readout["freeze"],
        target_audit=study.artifact(rank.OUT/"target_audit.json"),
        code=[study.artifact(Path(x)) for x in (__file__, relative.__file__, pair.__file__)],
        reference="additive linear ridge on the original natural examples; donor reference leaves out donor, calibration and held",
        labels="known relative gain and missing-grid debit on risk-feasible donor slates noninferior to reference",
        inference="freeze prior feasibility scores; compare top one to best positive pair lower prediction within top eight",
        sites=SITES, seeds=SEEDS, steps=STEPS, checkpoint_every=EVERY, label_limit=LABEL_LIMIT,
        shortlist=SHORTLIST, modes=MODES, calibration="unchanged grid and 2 percent checks for both arms",
        no_new_split=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), cohorts, old


def linear_features(action):
    return np.column_stack((np.ones(len(action)), action.sum(1))).astype(float)


def ridge_reference(examples, names):
    gram, rhs = None, None
    contributing = []
    for s in names:
        if (v := examples[s]) is None:
            continue
        x = linear_features(v[0].numpy())
        y = v[-1][:, 0].numpy()
        a, b = x.T@x/len(x), x.T@y/len(x)
        gram, rhs = (a, b) if gram is None else (gram+a, rhs+b)
        contributing.append(s)
    if not contributing:
        raise ValueError("No lawful additive-reference fitting inputs")
    beta = np.linalg.solve(gram/len(contributing)+.01*np.eye(len(rhs)), rhs/len(contributing))
    return beta, contributing


def reference(p, beta):
    x, _ = pair.encode(p, p["opts"])
    i = int((linear_features(x)@beta).argmax())
    return pair.paths_for(p, p["opts"][i:i+1])[0]


def conditional_examples(inputs, truth, cut, beta):
    rows, queries = [], 0
    rng = np.random.default_rng(6601)
    for ix, p in pair.menus(inputs):
        ref = reference(p, beta)
        y, v = truth["y"][ix], truth["v"][ix]
        eligible = []
        for k in range(0, len(p["opts"]), 128):
            paths = pair.paths_for(p, p["opts"][k:k+128])
            good = pair.labels(p, paths, y, v, cut)[:, 0] > 0.
            gain = np.asarray([target.lower(ref, z, y, v).sum() for z in paths])
            eligible.extend((np.flatnonzero(good & (gain >= -1e-10))+k).tolist())
        if not eligible:
            continue
        queries += 1
        take = np.sort(rng.choice(eligible, min(len(eligible), LABEL_LIMIT), replace=False))
        paths = pair.paths_for(p, p["opts"][take])
        x, xr, bound, unit, _ = relative.features(p["h"], p["c"], p["ols"], paths, ref)
        label = relative.labels(paths, ref, y, v, unit)
        rows.append((x, xr, bound, label))
    if not rows:
        return None, queries
    return tuple(torch.from_numpy(np.concatenate(r)) for r in zip(*rows, strict=True)), queries


def fit(key, held, cal, seed, donors, binding):
    name = f"{held}_{seed}"
    path = LOCAL/f"{name}.pt"
    identity = dict(registration=key, held=held, cal=cal, seed=seed, donors=binding)
    torch.manual_seed(seed)
    net = relative.RelativeGain()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+277)
    start, trace = 0, []
    if path.exists():
        c = torch.load(path, map_location="cpu", weights_only=True)
        if c["identity"] != identity or c["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Incorrect resume state")
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["optimizer"])
        start, trace, rng.bit_generator.state = c["step"], c["trace"], c["rng"]
    data = tuple(donors.values())
    for step in range(start+1, STEPS+1):
        counts = rng.multinomial(BATCH, np.full(len(data), 1./len(data)))
        indices = [rng.integers(len(d[0]), size=n) for d, n in zip(data, counts, strict=True)]
        x, xr, b, y = [torch.cat([d[k][i] for d, i in zip(data, indices, strict=True)]) for k in range(4)]
        loss = torch.nn.functional.smooth_l1_loss(net(x, xr, b), y, beta=.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(path)
            beat(phase="fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(path), trace=trace)


@torch.inference_mode()
def infer(gain, feasible, inp, beta):
    cv = inp["cv"]
    p = {k: cv.copy() for k in ("forced_add", *MODES)}
    scores = {k: np.full(len(cv), -1.) for k in MODES}
    selected = Counter()
    for ix, m in pair.menus(inp):
        a = reference(m, beta)
        p["forced_add"][ix] = a
        results = []
        for k in range(0, len(m["opts"]), 256):
            x, g = pair.encode(m, m["opts"][k:k+256])
            results.append(feasible(torch.from_numpy(x), torch.from_numpy(g)).sigmoid().numpy()[:, 1, 0])
        values = np.concatenate(results)
        shortlist = np.argsort(-values, kind="stable")[:SHORTLIST]
        candidates = pair.paths_for(m, m["opts"][shortlist])
        x, xr, b, _, _ = relative.features(m["h"], m["c"], m["ols"], candidates, a)
        prediction = gain(*(torch.from_numpy(z) for z in (x, xr, b))).numpy()
        score = prediction[:, 0]-prediction[:, 1]
        best = int(score.argmax()) if score.max() > 0. else 0
        for arm, i in (("feasibility", 0), ("pair_gain", best)):
            p[arm][ix] = candidates[i]
            scores[arm][ix] = values[shortlist[i]]
        selected["queries"] += 1
        selected["different"] += int(best > 0 and not np.allclose(candidates[best], candidates[0], atol=1e-9, rtol=1e-9))
        for arm in p:
            rr = pair.run.turn.turn.peak(p[arm][ix]-cv[ix])
            np.testing.assert_allclose(np.sort(rr[rr > 0.]), np.sort(m["radius"]), atol=1e-8, rtol=1e-8)
    return dict(q=inp["q"], cv=cv, ols4=inp["ols"], **p, **{f"score_{k}": v for k, v in scores.items()}), dict(selected)


def as_policy(p, mode):
    return dict(q=p["q"], cv=p["cv"], forced_add=p["forced_add"], forced_joint=p[mode],
                joint_probability=p[f"score_{mode}"])


def freeze(key, cohorts, old):
    src = pair.run.turn.exp.loader(cohorts)
    inputs = {s: pair.causal(cohorts[s]) for s in SITES}
    folds = {}
    for held in SITES:
        cal, train = pair.roles(held)
        cut = np.quantile(np.concatenate([src[s]["e"] for s in train]), [.25, .75])
        # A donor's reference uses other fitting sites only. Those reference
        # labels also exclude the donor from their cutoff estimates.
        for_main = {}
        for s in train:
            subcut = float(np.quantile(np.concatenate([src[d]["e"] for d in train if d != s]), .25))
            for_main[s] = pair.examples(inputs[s], src[s]["y"], src[s]["v"], subcut)
        beta, ref_sites = ridge_reference(for_main, train)
        donors, info = {}, {}
        for donor in train:
            inner = [s for s in train if s != donor]
            inner_examples = {}
            for s in inner:
                subcut = float(np.quantile(np.concatenate([src[d]["e"] for d in inner if d != s]), .25))
                inner_examples[s] = pair.examples(inputs[s], src[s]["y"], src[s]["v"], subcut)
            donor_beta, used = ridge_reference(inner_examples, inner)
            easy = float(np.quantile(np.concatenate([src[s]["e"] for s in inner]), .25))
            examples, queries = conditional_examples(inputs[donor], src[donor], easy, donor_beta)
            info[donor] = dict(source=cohorts[donor], reference_sites=used, donor_reference=donor_beta.tolist(),
                               cut=easy, queries=queries, slates=len(examples[0]) if examples is not None else 0)
            if examples is not None:
                donors[donor] = examples
        for seed in SEEDS:
            name = f"{held}_{seed}"
            cache = LOCAL/f"{name}.json"
            parent = old["folds"][f"{name}_matched_rank"]
            if parent["cut"] != cut.tolist():
                raise ValueError("Frozen feasibility reference changed")
            if not cache.exists():
                g, fit_info = fit(key, held, cal, seed, donors, info)
                feasibility = pair.model.PairAdmission().eval()
                record = parent["fit"]
                state = torch.load(study.checked(record["checkpoint"]), map_location="cpu", weights_only=True)
                ident = state["identity"]
                if (ident != record["identity"] or state["step"] != rank.STEPS or ident["held"] != held
                        or ident["cal"] != cal or not set(ident["donors"]) <= set(train)):
                    raise ValueError("Wrong frozen feasibility fold")
                feasibility.load_state_dict(state["net"])
                result = {}
                for role, s in (("cal", cal), ("held", held)):
                    p, counts = infer(g, feasibility, inputs[s], beta)
                    with np.load(study.checked(parent[role]), allow_pickle=False) as z:
                        np.testing.assert_array_equal(p["feasibility"], z["forced_joint"])
                        np.testing.assert_array_equal(p["score_feasibility"], z["joint_probability"])
                    thresholds = {}
                    if role == "cal":
                        thresholds = {m: pair.choose_threshold(as_policy(p, m), src[cal], cut)[0] for m in MODES}
                    pth = pair.run.turn.exp.previous.save_bank(LOCAL/f"{name}_{role}.npz", p)
                    result[role] = dict(forecast=pth, counts=counts, thresholds=thresholds)
                base.immutable(cache, dict(registration=key, fit=fit_info, cut=cut.tolist(), reference=beta.tolist(),
                    reference_sites=ref_sites, feasibility=parent["fit"]["checkpoint"], **result))
            saved = study.read(cache)
            if saved["registration"] != key or saved["fit"]["identity"]["donors"] != info:
                raise ValueError("Stored role split changed")
            for role in ("held", "cal"):
                study.checked(saved[role]["forecast"])
            folds[name] = saved
            beat(phase="freeze", fold=name, done=len(folds), total=len(SITES)*len(SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=folds, held_outcomes_used=False))


def evaluate(key, cohorts):
    path = OUT/"freeze.json"
    f = study.read(path)
    if f["registration"] != key or f["held_outcomes_used"] or set(f["folds"]) != {f"{s}_{r}" for s in SITES for r in SEEDS}:
        raise ValueError("Require complete held prediction freeze")
    data = pair.run.turn.exp.loader(cohorts)
    inputs = {s: pair.causal(cohorts[s]) for s in SITES}
    rows, sites = ({m: [] for m in MODES} for _ in range(2))
    held_counts = Counter()
    opportunities, thresholds = {}, {}
    for s in SITES:
        loc = {m: [] for m in MODES}
        for seed in SEEDS:
            name = f"{s}_{seed}"
            item = f["folds"][name]
            cal = item["fit"]["identity"]["cal"]
            cut = item["cut"]
            opportunities[name], thresholds[name] = {}, {}
            for role, site in (("cal", cal), ("held", s)):
                d = data[site]
                p = rank.checked_npz(item[role]["forecast"])
                opportunities[name][role] = {}
                for m in MODES:
                    policy = as_policy(p, m)
                    opportunities[name][role][m] = target.audit(inputs[site], d, cut[0], policy)
                    if role == "cal":
                        thresholds[name][m] = dict(grid=item["cal"]["thresholds"][m],
                            **rank.diagnostic.exact_thresholds(policy, d, cut))
                    else:
                        t = item["cal"]["thresholds"][m]
                        pred = {**{k: policy[k] for k in ("cv", "forced_add", "forced_joint")}, "ols4": p["ols4"],
                                **pair.release(policy, t)}
                        loc[m].extend(pair.assess(pred, d, cut))
                if role == "held":
                    held_counts.update(item[role]["counts"])
            beat(phase="score", fold=name)
        for m in MODES:
            sites[m].append(pair.summarize(loc[m]))
            rows[m].extend({**r, "record": f"{s}:{r['record']}"} for r in loc[m])
    totals = {m: rank.metrics(rows[m], sites[m]) for m in MODES}
    if totals["feasibility"]["forced_add"]["edits"] != totals["pair_gain"]["forced_joint"]["edits"]:
        raise ValueError("Forced budgets not matched")
    captures = {role: {m: dict(sum((Counter(r[role][m]) for r in opportunities.values()), Counter())) for m in MODES}
                for role in ("cal", "held")}
    out = dict(registration=key, freeze=study.artifact(path), totals=totals, thresholds=thresholds, captures=captures,
        captures_by_fold=opportunities, count_query_changes=dict(held_counts),
        training_fits=len(f["folds"]), independent_confirmation=False, reserved_outcomes_opened=False,
        original_2pct=True, result_source="fresh_run")
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps(dict(captures=captures, decisions=out["count_query_changes"]), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for path in (OUT, LOCAL):
        path.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, old = register()
        if phase == "freeze":
            freeze(key, cohorts, old)
        else:
            evaluate(key, cohorts)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
