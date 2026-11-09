"""Separate hard-negative mining from within-query ordering on OpenDD."""

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
from scripts import audit_m3w_atomic_pair_calibration as diagnostic

pair = diagnostic.pair
np, torch, study, base = pair.np, pair.torch, pair.study, pair.base
SITES, SEEDS = pair.SITES, pair.SEEDS
OUT, LOCAL = pair.OUT.parent/"opendd_pair_ranking_v1", pair.LOCAL.parent/"opendd_pair_ranking_v1"
METHODS = ("mined_bce", "matched_rank")
STEPS, EVERY, BATCH, PAIR_BATCH = pair.STEPS, pair.EVERY, pair.BATCH, 32
F = torch.nn.functional


def beat(**info):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(pid=os.getpid(), epoch=time(), **info))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, cohorts = pair.register()
    before_path = pair.OUT/"summary.json"
    before = study.read(before_path)
    old = study.read(study.checked(before["freeze"]))
    if before["registration"] != key or old["registration"] != key or old["held_targets_used"]:
        raise ValueError("Require the role-separated pointwise control")
    p = OUT/"registration.json"
    base.immutable(p, dict(parent=study.artifact(before_path), parent_freeze=before["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, pair.__file__, diagnostic.__file__)],
        methods=METHODS, roles={s: pair.roles(s) for s in SITES}, seeds=SEEDS, steps=STEPS, batch=BATCH,
        pair_batch=PAIR_BATCH, checkpoint_every=EVERY, workers=0, threads=4,
        mining="donor only, up to four positives and two geometrically nearest negatives per positive; prefer negative with easy support",
        loss="shared natural BCE + 0.25 mined BCE; matched_rank additionally 0.5 pairwise logistic on strict feasibility logits",
        selection="same candidate menu, exact radii, thresholds and original recording criteria as parent",
        calibration_diagnostic="all nonempty breakpoints after freeze, never applied to held",
        original_2pct=True, no_new_split=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), cohorts, old


def mine(actions, geom, labels, rng):
    good = np.flatnonzero(labels[:, 0] == 1.)
    bad = np.flatnonzero(labels[:, 0] == 0.)
    if not len(good) or not len(bad):
        return np.empty((0, 2), int)
    with_support = bad[labels[bad, 1] == 1.]
    bad = with_support if len(with_support) else bad
    good = np.sort(rng.choice(good, min(4, len(good)), replace=False))
    sym = np.column_stack((actions.sum(1), np.abs(actions[:, 0]-actions[:, 1]), geom))
    dist = np.square(sym[good, None]-sym[None, bad]).mean(2)
    near = bad[np.argsort(dist, axis=1, kind="stable")[:, :2]]
    return np.column_stack((np.repeat(good, near.shape[1]), near.ravel()))


def examples(inp, truth, cut):
    natural, mined = [], []
    rng, extra_rng = np.random.default_rng(2187), np.random.default_rng(3119)
    for ix, p in pair.menus(inp):
        opts = p["opts"]
        x, g = pair.encode(p, opts)
        labels = np.concatenate([pair.labels(p, pair.paths_for(p, opts[k:k+128]),
                                             truth["y"][ix], truth["v"][ix], cut)
                                 for k in range(0, len(opts), 128)])
        take = np.sort(rng.choice(len(opts), min(pair.LIMIT, len(opts)), replace=False))
        natural.append((x[take], g[take], labels[take]))
        pairs = mine(x, g, labels, extra_rng)
        if len(pairs):
            mined.append((x[pairs], g[pairs], labels[pairs]))
    def tensors(rows):
        return tuple(torch.from_numpy(np.concatenate(v)) for v in zip(*rows, strict=True)) if rows else None
    return tensors(natural), tensors(mined)


def bce(logits, y):
    return (F.binary_cross_entropy_with_logits(logits, y.unsqueeze(-2).expand_as(logits), reduction="none")
            * logits.new_tensor([1., .25])).mean()


def ranking_loss(ordered):
    """Both heads should score the same-query positive above its negative."""
    return F.softplus(ordered[:, 1, :, 0]-ordered[:, 0, :, 0]).mean()


def draw(rng, sources, n):
    counts = rng.multinomial(n, np.full(len(sources), 1./len(sources)))
    ix = [rng.integers(len(d[0]), size=k) for d, k in zip(sources, counts, strict=True)]
    return tuple(torch.cat([d[j][i] for d, i in zip(sources, ix, strict=True)]) for j in range(3))


def fit(key, held, cal, seed, method, donor, binding):
    name = f"{held}_{seed}_{method}"
    identity = dict(registration=key, held=held, cal=cal, seed=seed, method=method, donors=binding)
    path = LOCAL/f"{name}.pt"
    torch.manual_seed(seed)
    net = pair.model.PairAdmission()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+827)
    start, trace = 0, []
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=True)
        if old["identity"] != identity or old["step"] not in range(EVERY, STEPS+1, EVERY):
            raise ValueError("Resume roles or model changed")
        net.load_state_dict(old["net"])
        opt.load_state_dict(old["optimizer"])
        start, trace, rng.bit_generator.state = old["step"], old["trace"], old["rng"]
    natural = [v[0] for v in donor.values() if v[0] is not None]
    mined = [v[1] for v in donor.values() if v[1] is not None]
    if not natural or not mined:
        raise ValueError("No matched donor negatives available")
    for step in range(start+1, STEPS+1):
        x, g, y = draw(rng, natural, BATCH)
        a, b, z = draw(rng, mined, PAIR_BATCH)
        ordered = net(a.flatten(0, 1), b.flatten(0, 1)).reshape(len(a), 2, 2, 2)
        rank = ranking_loss(ordered)
        common = bce(net(x, g), y)+.25*bce(ordered, z)
        loss = common+.5*rank if method == "matched_rank" else common
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % EVERY == 0:
            trace.append(dict(step=step, common=float(common.detach()), pairwise=float(rank.detach())))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(path)
            beat(phase="fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(identity=identity, checkpoint=study.artifact(path), trace=trace)


def freeze(key, cohorts):
    src = pair.run.turn.exp.loader(cohorts)
    inputs = {s: pair.causal(cohorts[s]) for s in SITES}
    fold = {}
    for held in SITES:
        cal, train = pair.roles(held)
        cut = np.quantile(np.concatenate([src[s]["e"] for s in train]), [.25, .75])
        donors, binding = {}, {}
        for donor in train:
            donor_cut = float(np.quantile(np.concatenate([src[s]["e"] for s in train if s != donor]), .25))
            a, b = examples(inputs[donor], src[donor], donor_cut)
            if a is not None:
                donors[donor] = (a, b)
                binding[donor] = dict(source=cohorts[donor], cut=donor_cut, natural=len(a[0]),
                                     matched_pairs=len(b[0]) if b is not None else 0)
        for seed in SEEDS:
            for method in METHODS:
                name = f"{held}_{seed}_{method}"
                p = LOCAL/f"{name}.json"
                if not p.exists():
                    net, fit_info = fit(key, held, cal, seed, method, donors, binding)
                    cal_path = pair.predict(net, inputs[cal])
                    threshold, table = pair.choose_threshold(cal_path, src[cal], cut)
                    held_path = pair.predict(net, inputs[held])
                    held_path.update(pair.release(held_path, threshold))
                    save = pair.run.turn.exp.previous.save_bank
                    base.immutable(p, dict(registration=key, fit=fit_info, cut=cut.tolist(), threshold=threshold,
                        calibration_table=table, cal=save(LOCAL/f"{name}_cal.npz", cal_path),
                        held=save(LOCAL/f"{name}_held.npz", held_path)))
                item = study.read(p)
                ident = item["fit"]["identity"]
                if (item["registration"] != key or ident["cal"] != cal or ident["held"] != held
                        or ident["seed"] != seed or ident["method"] != method or not set(ident["donors"]) <= set(train)):
                    raise ValueError("Role overlap or old fold")
                for ref in (item["cal"], item["held"], item["fit"]["checkpoint"]):
                    study.checked(ref)
                fold[name] = item
                beat(phase="freeze", fold=name, done=len(fold), total=2*len(SITES)*len(SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=fold, held_outcomes_used=False))


def captures(inputs, truth, cut, predictions):
    result = []
    for ix, p in pair.menus(inputs):
        possible = any(pair.labels(p, pair.paths_for(p, p["opts"][k:k+128]),
                                   truth["y"][ix], truth["v"][ix], cut)[:, 0].any()
                       for k in range(0, len(p["opts"]), 128))
        selected = {m: bool(pair.labels(p, y["forced_joint"][ix][None],
                                        truth["y"][ix], truth["v"][ix], cut)[0, 0])
                    for m, y in predictions.items()}
        if any(selected.values()) and not possible:
            raise ValueError("Frozen prediction outside same candidate menu")
        result.append(dict(record=str(truth["rec"][ix.start]), opportunity=possible, selected=selected))
    return result


def metrics(rows, sites):
    total = pair.summarize(rows)
    for a in pair.ARMS:
        total[a]["location_ci_gain_pp"] = pair.run.turn.effects.interval([s[a]["gain"] for s in sites])
    for k in ("forced", "cal"):
        total[k]["location_ci_pair_native"] = pair.run.turn.effects.interval([s[k]["site_mean_native"] for s in sites])
        if total[f"{k}_add"]["edits"] != total[f"{k}_joint"]["edits"]:
            raise ValueError("Changed matched budget")
    return {k: v for k, v in total.items() if k not in ("cv", "ols4")}


def checked_npz(ref):
    with np.load(study.checked(ref), allow_pickle=False) as z:
        return dict(z)


def score(key, cohorts, old):
    path = OUT/"freeze.json"
    frozen = study.read(path)
    expected = {f"{s}_{i}_{m}" for s in SITES for i in SEEDS for m in METHODS}
    if frozen["registration"] != key or frozen["held_outcomes_used"] or set(frozen["folds"]) != expected:
        raise ValueError("Require complete sealed comparison")
    src = pair.run.turn.exp.loader(cohorts)
    causal = {s: pair.causal(cohorts[s]) for s in SITES}
    cal_scores, held_scores, thresholds = {}, {}, {}
    all_rows = {m: [] for m in METHODS}
    sites = {m: [] for m in METHODS}
    for held in SITES:
        cal, _ = pair.roles(held)
        local = {m: [] for m in METHODS}
        for seed in SEEDS:
            name = f"{held}_{seed}"
            fit = {m: frozen["folds"][f"{name}_{m}"] for m in METHODS}
            cut = fit["matched_rank"]["cut"]
            if cut != fit["mined_bce"]["cut"] or cut != old["calibration"][name]["cut"]:
                raise ValueError("Scoring reference changed")
            held_pred = {m: checked_npz(a["held"]) for m, a in fit.items()}
            held_pred["original_bce"] = checked_npz(old["forecasts"][name])
            for k, x in (("q", src[held]["q"]), ("cv", src[held]["cv"]), ("ols4", src[held]["ols"])):
                for z in held_pred.values():
                    np.testing.assert_array_equal(z[k], x)
            q = src[held]["q"]
            ref = pair.run.turn.turn.peak(held_pred["original_bce"]["forced_joint"]-src[held]["cv"])
            for m in METHODS:
                pred = held_pred[m]
                for arm in ("forced_add", "forced_joint"):
                    radius = pair.run.turn.turn.peak(pred[arm]-src[held]["cv"])
                    np.testing.assert_allclose(radius[np.lexsort((radius, q))], ref[np.lexsort((ref, q))], rtol=1e-8, atol=1e-8)
                local[m].extend(pair.assess({k: pred[k] for k in ("cv", "ols4", *pair.ARMS)}, src[held], cut))
            cap = captures(causal[held], src[held], cut[0], held_pred)
            held_scores[name] = {m: sum(r["selected"][m] for r in cap) for m in held_pred}
            held_scores[name].update(opportunities=sum(r["opportunity"] for r in cap), queries=len(cap), site=held)
            cal_pred = {m: checked_npz(a["cal"]) for m, a in fit.items()}
            cap = captures(causal[cal], src[cal], cut[0], cal_pred)
            cal_scores[name] = {m: sum(r["selected"][m] for r in cap) for m in METHODS}
            cal_scores[name].update(opportunities=sum(r["opportunity"] for r in cap), queries=len(cap), site=cal)
            thresholds[name] = {m: dict(grid=fit[m]["threshold"], **diagnostic.exact_thresholds(z, src[cal], cut))
                                for m, z in cal_pred.items()}
            beat(phase="score", fold=name)
        for m in METHODS:
            sites[m].append(pair.summarize(local[m]))
            all_rows[m].extend({**r, "record": f"{held}:{r['record']}"} for r in local[m])
    totals = {m: metrics(all_rows[m], sites[m]) for m in METHODS}
    method_diff = {}
    for name, data in (("calibration", cal_scores), ("held", held_scores)):
        cohorts = {s: [v for v in data.values() if v["site"] == s] for s in SITES}
        method_diff[name] = pair.run.turn.effects.interval([
            100.*sum(v["matched_rank"]-v["mined_bce"] for v in rows)/n
            for rows in cohorts.values() if (n := sum(v["opportunities"] for v in rows))])
    out = dict(registration=key, freeze=study.artifact(path), result_source="fresh_run",
        totals=totals, fit_count=len(frozen["folds"]), held_capture=held_scores, calibration_capture=cal_scores,
        thresholds=thresholds, capture_difference_rank_minus_mined_pp=method_diff, original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps(dict(capture_effect=method_diff, thresholds=Counter(
        (m, a["grid"] is not None) for x in thresholds.values() for m, a in x.items()).most_common()), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, old = register()
        if phase == "freeze":
            freeze(key, cohorts)
        else:
            score(key, cohorts, old)
        beat(phase=phase+"_complete")


if __name__ == "__main__":
    main()
