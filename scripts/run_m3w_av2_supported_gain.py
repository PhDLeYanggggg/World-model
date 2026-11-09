"""Separate conditional gain from eligibility on the exact AV2 two-edit menu."""

import argparse
from collections import Counter
import fcntl
from math import log
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_pair_refit as prev

av, np, torch, study, base = prev.av, prev.np, prev.torch, prev.study, prev.base
OUT, LOCAL = prev.OUT.parent/"av2_supported_gain_v1", prev.LOCAL.parent/"av2_supported_gain_v1"
METHODS = ("conditional", "censored")
ARMS = ("cv", "ols4", "additive", "frozen", "signed_product", *METHODS, "calibrated")
F = torch.nn.functional


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, original, source, seal = prev.register()
    p = prev.OUT/"summary.json"
    old = study.read(p)
    freeze = study.read(study.checked(old["freeze"]))
    if (old["registration"] != parent or freeze["registration"] != parent or freeze["held_labels_used"]
            or old["reserved_outcomes_opened"] or old["independent_confirmation"]):
        raise ValueError("Require the sealed previous AV2 development run")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), previous_freeze=old["freeze"],
        code=[study.artifact(Path(x)) for x in (__file__, prev.__file__)],
        data="same 128 publisher-train scenarios, 8 observation / 12 prediction steps",
        matches="same four donors, one calibration city, one held city, three seeds, menu and reference",
        methods=METHODS, steps=prev.STEPS, batch=prev.BATCH, checkpoint_every=prev.SAVE,
        conditional="squared loss only on qualifying options; p(strict) times conditional positive gain",
        censored="squared loss on gain for qualifying options and zero elsewhere; no probability multiplier",
        heads="same MLP, weighted strict BCE and support BCE as parent; log-odds correction for weight 5",
        calibration="unchanged tests over exact nonempty breakpoints; maximize calibration coverage then prefer conditional",
        diagnostics="forced decisions and strict option retrieval are not releases",
        cpu_threads=4, workers=0, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), original, source, seal, old, freeze


def gain_loss(logit, bound, gain, strict, method):
    value = logit.sigmoid()*bound
    if method == "conditional":
        # No negative examples define the magnitude of a qualifying gain.
        return ((value-gain).square()*strict).sum()/strict.sum().clamp_min(1)
    if method == "censored":
        return F.mse_loss(value, gain*strict)
    raise ValueError("Unknown gain objective")


def scores(z, bound, method):
    value = z[:, 0].sigmoid()*bound
    if method == "conditional":
        return value*(z[:, 1]-log(5.)).sigmoid()
    if method == "censored":
        return value
    raise ValueError("Unknown gain objective")


def fit(identity, examples):
    held, seed, method = (identity[k] for k in ("held", "seed", "method"))
    p = LOCAL/f"{held}_{seed}_{method}.pt"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+5507)
    net = prev.network()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    start, losses = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != identity or ck["step"] not in range(prev.SAVE, prev.STEPS+1, prev.SAVE):
            raise ValueError("Cannot resume a different fold or objective")
        start, losses = ck["step"], ck["loss"]
        rng.bit_generator.state = ck["rng"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    queries = tuple(examples.values())
    if not queries or not any(bool(x[3].any()) for x in queries):
        raise ValueError("No qualifying donor candidates")
    for step in range(start+1, prev.STEPS+1):
        counts = rng.multinomial(prev.BATCH, np.full(len(queries), 1./len(queries)))
        batch = [(x, rng.integers(len(x[0]), size=n)) for x, n in zip(queries, counts, strict=True) if n]
        x, b, g, strict, support = (torch.cat([z[k][i] for z, i in batch]) for k in range(5))
        pred = net(x)
        lg = gain_loss(pred[:, 0], b, g, strict, method)
        ls = F.binary_cross_entropy_with_logits(pred[:, 1], strict, pos_weight=torch.tensor(5.))
        la = F.binary_cross_entropy_with_logits(pred[:, 2], support)
        loss = lg+ls+.25*la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % prev.SAVE == 0:
            losses.append([float(t.detach()) for t in (loss, lg, ls, la)])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, loss=losses), tmp)
            tmp.replace(p)
            beat("fit", fold=f"{held}_{seed}", method=method, step=step, total=prev.STEPS)
    return net.eval(), study.artifact(p), losses


@torch.inference_mode()
def choose(net, d, seed, method):
    a, j, paths, feat = prev.candidates(d, seed)
    if feat is None:
        return j, -1., np.empty(0, dtype=int)
    x, b, _ = feat
    s = scores(net(torch.from_numpy(x)), torch.from_numpy(b), method).numpy()
    order = np.argsort(-s, kind="stable")[:10]
    order = order[s[order] > 0.]
    return (paths[order[0]], float(s[order[0]]), order) if len(order) else (j, -1., order)


def calibration(rows):
    threshold, tested, feasible = prev.calibrate(rows)
    return dict(threshold=threshold, tested=tested, feasible=feasible,
        local_strict=sum(r["strict"] for r in rows),
        strict_with_add=sum(r["strict"] and r["add_ok"] for r in rows),
        accepted=sum(r["score"] >= threshold for r in rows) if threshold is not None else 0)


def winner(table):
    positive = [m for m in METHODS if table[m]["accepted"]]
    return max(positive, key=lambda m: table[m]["accepted"]) if positive else None


def train(key, original, source, seal, parent):
    data, done = prev.read_inputs(original, seal), {}
    cities = sorted({d["city"] for d in data.values()})
    for held in cities:
        cal, donors = prev.roles(cities, held)
        names = lambda subset: [n for n, d in data.items() if d["city"] in subset]
        donor_y = prev.load_targets(source, data, names(donors))
        cal_y = prev.load_targets(source, data, names([cal]))
        cut = prev.donor_cut(data, donor_y)
        held_names = names([held])
        for seed in prev.SEEDS:
            fold = f"{held}_{seed}"
            prior = parent["folds"][fold]
            roles = dict(held=held, calibration=cal, donors=donors, seed=seed)
            if any(prior["identity"][k] != v for k, v in roles.items()):
                raise ValueError("Parent roles changed")
            np.testing.assert_array_equal(cut, prior["cut"])
            receipt = LOCAL/f"{fold}.json"
            ident = dict(registration=key, **roles)
            if not receipt.exists():
                examples = prev.examples(data, donor_y, seed, cut)
                np.testing.assert_equal(
                    (len(examples), sum(len(x[0]) for x in examples.values()), sum(x[3].sum().item() for x in examples.values())),
                    (prior["donor_queries"], prior["donor_options"], prior["donor_strict"]))
                fits, table, nets = {}, {}, {}
                for method in METHODS:
                    nets[method], checkpoint, loss = fit(dict(ident, method=method), examples)
                    fits[method] = dict(checkpoint=checkpoint, loss=loss)
                    rows = []
                    for name, (y, v) in cal_y.items():
                        d = data[name]
                        path, s, _ = choose(nets[method], d, seed, method)
                        if s <= 0.:
                            continue
                        a = d[f"additive_{seed}"]
                        _, strict, _ = prev.label_terms(d, path[None], a, y, v, cut)
                        ref = prev.metrics(a, d, y, v, cut)
                        add_ok = ref["risk"]["supported"] and ref["gain"] is not None and min(ref["gain"]) > 0.
                        rows.append(dict(score=s, strict=bool(strict[0]), add_ok=add_ok))
                    table[method] = calibration(rows)
                forecasts = {}
                for n in held_names:
                    values = dict(registration=key, seed=seed)
                    for m in METHODS:
                        path, s, order = choose(nets[m], data[n], seed, m)
                        values.update({m: path, f"score_{m}": s, f"order_{m}": order})
                    forecasts[n] = av.turn.exp.previous.save_bank(LOCAL/f"{n}_{seed}.npz", values)
                base.immutable(receipt, dict(identity=ident, fits=fits, forecasts=forecasts,
                    cal=table, chosen=winner(table), cut=cut.tolist()))
            z = study.read(receipt)
            if z["identity"] != ident or set(z["forecasts"]) != set(held_names) or z["chosen"] != winner(z["cal"]):
                raise ValueError("Resumed fold disagrees with the registered roster or calibration")
            for ref in (*z["forecasts"].values(), *(v["checkpoint"] for v in z["fits"].values())):
                study.checked(ref)
            done[fold] = z
            beat("freeze", fold=fold, done=len(done), total=len(cities)*len(prev.SEEDS))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, original, source, seal, old, parent):
    p = OUT/"freeze.json"
    freeze = study.read(p)
    data = prev.read_inputs(original, seal)
    cities = sorted({d["city"] for d in data.values()})
    if (freeze["registration"] != key or freeze["held_labels_used"]
            or set(freeze["folds"]) != set(parent["folds"])):
        raise ValueError("Need all folds frozen before scoring")
    truth, rows, counts = prev.load_targets(source, data, data), [], Counter()
    for n, d in data.items():
        y, v = truth[n]
        for seed in prev.SEEDS:
            fold = f"{d['city']}_{seed}"
            z, prior = freeze["folds"][fold], parent["folds"][fold]
            cal, donors = prev.roles(cities, d["city"])
            if z["identity"] != dict(registration=key, held=d["city"], calibration=cal, donors=donors, seed=seed):
                raise ValueError("Score receipt does not match disjoint city roles")
            with np.load(study.checked(z["forecasts"][n]), allow_pickle=False) as raw:
                if raw["registration"].item() != key or raw["seed"].item() != seed:
                    raise ValueError("New forecasts have wrong registration")
                forecast = dict(raw)
            with np.load(study.checked(prior["forecasts"][n]), allow_pickle=False) as raw:
                if raw["registration"].item() != prior["identity"]["registration"] or raw["seed"].item() != seed:
                    raise ValueError("Old forecasts have wrong registration")
                signed = raw["refit"]
            a, j, paths, _ = prev.candidates(d, seed)
            c, o, cut = d["cv"], d["ols4"], z["cut"]
            method = z["chosen"]
            release = c
            if method is not None and forecast[f"score_{method}"].item() >= z["cal"][method]["threshold"]:
                release = forecast[method]
            pred = dict(zip(ARMS, (c, o, a, j, signed, *(forecast[m] for m in METHODS), release), strict=True))
            active = {}
            if paths is not None:
                _, strict, _ = prev.label_terms(d, paths, a, y, v, cut)
                active = dict(two_slots=1, oracle=int(strict.any()))
                for m in ("signed_product", *METHODS):
                    _, q, _ = prev.label_terms(d, pred[m][None], a, y, v, cut)
                    active[f"{m}_selected"] = int(q[0])
                    if m in METHODS:
                        order = forecast[f"order_{m}"]
                        if len(order):
                            np.testing.assert_array_equal(paths[order[0]], pred[m])
                        active[f"{m}_top5"] = int(strict[order[:5]].any())
                        active[f"{m}_top10"] = int(strict[order].any())
                counts.update(active)
                if v.all():
                    counts.update({f"complete_{k}": value for k, value in active.items()})
            terms = {m: {k: av.frontier.pair_reduction(pred[m][None], ref, y, v)[0].tolist()
                         for k, ref in (("vs_additive", a), ("vs_signed", signed), ("vs_frozen", j))}
                     for m in METHODS}
            rows.append(dict(clip=n, city=d["city"], seed=seed, complete=bool(v.all()), **active,
                             metrics={m: prev.metrics(vv, d, y, v, cut) for m, vv in pred.items()}, terms=terms))
        beat("score", done=len(rows)//len(prev.SEEDS), total=len(data))
    totals = {}
    for m in ARMS:
        values = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        totals[m] = dict(edits=sum(x["selected"] for x in values), edited=len(values),
            risk_pass=sum(x["supported"] for x in values),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in values),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in values))
        totals[m].update({k: av.interval(rows, lambda r: r["metrics"][m][k])
                          for k in ("ade", "fde", "gain", "hard", "tail")})
    if totals["signed_product"] != old["metrics"]["refit"] or totals["frozen"] != old["metrics"]["frozen"]:
        raise ValueError("Archived comparison did not replay")
    if len({totals[m]["edits"] for m in ("additive", "frozen", "signed_product", *METHODS)}) != 1:
        raise ValueError("Forced edit count changed")
    if counts["oracle"] != old["capacity"]["oracle"] or counts["complete_oracle"] != old["capacity"]["complete_oracle"]:
        raise ValueError("Matched oracle did not replay")
    contrasts = {m: {k: {g: av.interval([r for r in rows if g == "all" or r["complete"]],
                                       lambda r: r["terms"][m][k]) for g in ("all", "complete")}
                     for k in ("vs_additive", "vs_signed", "vs_frozen")} for m in METHODS}
    calib = {m: dict(sum((Counter({k: v for k, v in z["cal"][m].items() if k != "threshold"})
                              for z in freeze["folds"].values()), Counter())) for m in METHODS}
    out = dict(registration=key, freeze=study.artifact(p), source="fresh_run", fits=len(freeze["folds"])*len(METHODS),
        metrics=totals, contrasts=contrasts, capacity=dict(counts), calibration=calib,
        selected_methods=dict(Counter(z["chosen"] or "cv" for z in freeze["folds"].values())),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(LOCAL/"rows.json", rows)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("capacity", "calibration", "selected_methods")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, original, source, seal, old, parent = register()
        if phase == "fit":
            train(key, original, source, seal, parent)
        else:
            score(key, original, source, seal, old, parent)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
