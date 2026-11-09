"""Test within-query ranking with and without observed-history shape replay."""

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
from scripts import run_m3w_av2_oof_selection as old
from src.world_model import m3w_av2_history_replay as history

av, prev, eq, np, torch, study, base = old.av, old.prev, old.eq, old.np, old.torch, old.study, old.base
OUT, LOCAL = old.OUT.parent/"av2_history_rank_v1", old.LOCAL.parent/"av2_history_rank_v1"
STEPS, SAVE, BATCH, PAIRS = 320, 40, 128, 32
METHODS = ("geometry", "history")
ARMS = ("cv", "ols4", "raw_eq", "uniform", "additive", "prior_risk", "prior_interaction",
        *METHODS, "released_additive", "released_joint")
F = torch.nn.functional


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, src, roles, inner, outer, _ = old.register()
    p = old.OUT/"summary.json"
    previous = study.read(p)
    seal = study.read(study.checked(previous["freeze"]))
    if previous["registration"] != parent or seal["registration"] != parent or seal["held_labels_used"] or previous["reserved_outcomes_opened"]:
        raise ValueError("Require the complete OOF comparison")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(parent=study.artifact(p), parent_freeze=previous["freeze"],
        code=[study.artifact(Path(p)) for p in (__file__, history.__file__, old.__file__)],
        data="unchanged OOF candidates, six held cities, 896 added development scenarios, eight observed steps",
        ablation="same network, seed, training draws and loss; geometry supplies zeros for replay channels",
        history="align final-origin deformation to prefixes 4/5/6/7 and compare on observed suffix; not a rolling EqMotion forecast",
        target="absolute dual floor gain and original risk qualification; same-query sampled positive/negative ranking",
        reference="reuse parent learned additive and uniform decisions exactly",
        calibration="original both-arm qualification and conservative pair advantage over the unchanged additive; max coverage",
        steps=STEPS, checkpoint_every=SAVE, batch=BATCH, rank_pairs=PAIRS, seeds=eq.SEEDS,
        original_2pct=True, cpu_threads=4, workers=0, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, src, roles, inner, outer, seal, previous


def network():
    n = torch.nn
    return n.Sequential(n.Linear(old.credit.WIDTH+history.WIDTH, 64), n.SiLU(), n.Linear(64, 32),
                        n.SiLU(), n.Linear(32, 3))


def with_history(x, h, method):
    return torch.cat((x, h if method == "history" else torch.zeros_like(h)), dim=-1)


def expected(z, bound):
    return (z[:, 1]-log(5.)).sigmoid()*z[:, 0].sigmoid()*bound


def rank_loss(pos, neg):
    return F.softplus(neg.clamp_min(1e-8).log()-pos.clamp_min(1e-8).log()).mean()


def example(d, m, y, v, cut):
    x, g, _, value, _, strict, support = old.pair_targets(d, m, y, v, cut)
    p, rows, *_ = old.options(d, m, cut)
    hf = history.features(d["h"], d["cv"], p, rows, cut[0])
    return x, torch.from_numpy(hf), g, value, strict, support


def train_model(ident, examples, name):
    seed, method = ident["seed"], ident["method"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+9123)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    path = LOCAL/f"{name}_{method}.pt"
    step, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Checkpoint does not match the replay ablation")
        step, trace = ck["step"], ck["loss"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
    positive = [(z, np.flatnonzero(z[4]), np.flatnonzero(~z[4].bool())) for z in examples if z[4].any()]
    paired = [r for r in positive if len(r[2])]
    if not paired:
        raise ValueError("No same-query positive and negative training slates")

    def pull(sample):
        return [torch.cat([r[k][i] for r, i in sample]) for k in range(6)]

    for t in range(step+1, STEPS+1):
        sizes = rng.multinomial(BATCH, np.full(len(examples), 1./len(examples)))
        draws = [(z, rng.integers(len(z[0]), size=k)) for z, k in zip(examples, sizes, strict=True) if k]
        x, h, _, _, label, support = pull(draws)
        z = net(with_history(x, h, method))
        qualification = F.binary_cross_entropy_with_logits(z[:, 1], label, pos_weight=torch.tensor(5.))
        auxiliary = F.binary_cross_entropy_with_logits(z[:, 2], support)
        picked = [paired[i] for i in rng.integers(len(paired), size=PAIRS)]
        xp, hp, gp, value, _, _ = pull([(z, [int(rng.choice(a))]) for z, a, _ in picked])
        xn, hn, gn, _, _, _ = pull([(z, [int(rng.choice(b))]) for z, _, b in picked])
        qp, qn = net(with_history(xp, hp, method)), net(with_history(xn, hn, method))
        rank = rank_loss(expected(qp, gp), expected(qn, gn))
        magnitude = F.mse_loss(qp[:, 0].sigmoid()*gp, value)
        loss = qualification+.25*auxiliary+magnitude+.5*rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if t % SAVE == 0:
            trace.append([float(v.detach()) for v in (loss, qualification, magnitude, rank)])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=t, net=net.state_dict(), opt=opt.state_dict(),
                rng=rng.bit_generator.state, loss=trace), tmp)
            tmp.replace(path)
            beat("fit", fold=name, method=method, step=t, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), loss=trace,
                            donor_queries=len(examples), paired_donor_queries=len(paired))


def load_action(fold):
    ck = torch.load(study.checked(fold["action"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != dict(fold["identity"], model="action") or ck["step"] != old.STEPS:
        raise ValueError("Wrong learned additive reference")
    net = old.action.Action().eval()
    net.load_state_dict(ck["net"])
    return net


@torch.inference_mode()
def choose(models, reference, d, seed, cut, single):
    m = old.action.menu(d, seed, cut)
    if m is None:
        return single, dict.fromkeys(METHODS, single), dict.fromkeys(METHODS, -1.), {
            k: np.empty(0, dtype=int) for k in METHODS}
    p, rows, x, g, _, _ = old.options(d, m, cut)
    act = reference(torch.from_numpy(m["x"])).numpy()
    _, order, _ = old.action.rank(act, m["opts"], m["unit"][m["row"]])
    additive = p[order[0]]
    h = torch.from_numpy(history.features(d["h"], d["cv"], p, rows, cut[0]))
    x, g = torch.from_numpy(x), torch.from_numpy(g)
    out, confidence, top = {}, {}, {}
    for method, net in models.items():
        score = expected(net(with_history(x, h, method)), g).numpy()
        order = np.argsort(-score, kind="stable")[:5]
        order = order[score[order] > 0.]
        top[method] = order
        out[method], confidence[method] = ((p[order[0]], float(score[order[0]])) if len(order) else (additive, -1.))
    return additive, out, confidence, top


def freeze(key, data, source, roles, inner, outer, seal):
    receipts = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            k = f"{held}_{seed}"
            item, anchor = outer["folds"][k], seal["folds"][k]
            cut = anchor["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(ident[j] != anchor["identity"][j] for j in ("held", "calibration", "donors", "seed")):
                raise ValueError("Comparison roles changed")
            receipt = LOCAL/f"{k}.json"
            if not receipt.exists():
                reference = load_action(anchor)
                raw = old.donor_forecasts(inner, data, held, role, seed)
                if raw.keys() != dy.keys():
                    raise ValueError("Nested training rosters disagree")
                samples = []
                for n, (y, v) in dy.items():
                    d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
                    if (m := old.action.menu(d, seed, cut)) is not None:
                        samples.append(example(d, m, y, v, cut))
                models, fits = {}, {}
                for method in METHODS:
                    models[method], fits[method] = train_model(dict(ident, method=method), samples, k)
                raw, offsets = eq.predict(old.load_outer(item), data, list(cy))
                cal_rows = {m: [] for m in METHODS}
                for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
                    d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
                    y, v = cy[n]
                    a, pp, conf, _ = choose(models, reference, d, seed, cut, d["cv"])
                    ok = eq.ceiling.eligible_additive(prev.metrics(a, d, y, v, cut))
                    for m in METHODS:
                        if conf[m] > 0.:
                            _, q, _ = prev.label_terms(d, pp[m][None], a, y, v, cut)
                            cal_rows[m].append(dict(score=conf[m], strict=bool(q[0]), add_ok=ok))
                calibration = {m: old.action.prior.calibration(cal_rows[m]) for m in METHODS}
                selected = max(METHODS, key=lambda m: calibration[m]["accepted"])
                if not calibration[selected]["accepted"]:
                    selected = None
                with (np.load(study.checked(item["forecast"]), allow_pickle=False) as z,
                      np.load(study.checked(anchor["forecast"]), allow_pickle=False) as control):
                    np.testing.assert_array_equal(z["names"], control["names"])
                    np.testing.assert_array_equal(z["offsets"], control["offsets"])
                    out, scores, top = [{m: [] for m in METHODS} for _ in range(3)]
                    for n, lo, hi in zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True):
                        d = dict(data[n], bank=eq.expanded(data[n], z["raw_eq"][lo:hi]))
                        a, pp, conf, ranked = choose(models, reference, d, seed, cut, control["additive"][lo:hi])
                        np.testing.assert_array_equal(a, control["additive"][lo:hi])
                        for m in METHODS:
                            out[m].append(pp[m])
                            scores[m].append(conf[m])
                            top[m].append(np.pad(ranked[m], (0, 5-len(ranked[m])), constant_values=-1))
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{k}.npz", dict(names=z["names"], offsets=z["offsets"],
                        registration=key, **{m: np.concatenate(v) for m, v in out.items()},
                        **{f"score_{m}": np.asarray(v) for m, v in scores.items()},
                        **{f"top_{m}": np.asarray(v) for m, v in top.items()}))
                base.immutable(receipt, dict(identity=ident, fits=fits, forecast=bank, calibration=calibration, selected=selected))
            saved = study.read(receipt)
            if saved["identity"] != ident:
                raise ValueError("Mismatched forecast or calibration receipt")
            for x in (saved["forecast"], *(f["checkpoint"] for f in saved["fits"].values())):
                study.checked(x)
            receipts[k] = saved
            beat("freeze", fold=k, completed=len(receipts), total=len(seal["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=receipts, held_labels_used=False))


def score(key, data, source, outer, seal, prior):
    p = OUT/"freeze.json"
    frozen = study.read(p)
    if frozen["registration"] != key or frozen["held_labels_used"] or set(frozen["folds"]) != set(seal["folds"]):
        raise ValueError("Hold every forecast and threshold fixed before scoring")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows, counts = [], Counter()
    for k, item in frozen["folds"].items():
        held, seed, cut = (item["identity"][n] for n in ("held", "seed", "cut"))
        with (np.load(study.checked(item["forecast"]), allow_pickle=False) as a,
              np.load(study.checked(outer["folds"][k]["forecast"]), allow_pickle=False) as b,
              np.load(study.checked(seal["folds"][k]["forecast"]), allow_pickle=False) as control):
            for field in ("names", "offsets"):
                np.testing.assert_array_equal(a[field], b[field])
                np.testing.assert_array_equal(a[field], control[field])
            if a["registration"].item() != key or any(data[n]["city"] != held for n in a["names"]):
                raise ValueError("Wrong held fold")
            for i, (n, lo, hi) in enumerate(zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                c = d["cv"]
                pp = dict(cv=c, ols4=d["ols4"], raw_eq=b["raw_eq"][lo:hi], uniform=control["uniform"][lo:hi],
                    additive=control["additive"][lo:hi], prior_risk=control["risk_only"][lo:hi],
                    prior_interaction=control["interaction"][lo:hi], **{m: a[m][lo:hi] for m in METHODS})
                selected = item["selected"]
                active = selected is not None and a[f"score_{selected}"][i] >= item["calibration"][selected]["threshold"]
                pp.update(released_additive=pp["additive"] if active else c, released_joint=pp[selected] if active else c)
                metrics = {m: prev.metrics(f, d, y, v, cut) for m, f in pp.items()}
                ok = eq.ceiling.eligible_additive(metrics["additive"])
                if len(rr := eq.radii(d, seed)) == 2:
                    legal = eq.slates(d, rr, eq.expanded(d, pp["raw_eq"]))
                    _, strict, _ = prev.label_terms(d, legal, pp["additive"], y, v, cut)
                    counts.update(two_slots=1, additive_eligible=int(ok), any_strict=int(strict.any()),
                        paired_ceiling=int(ok and strict.any()), changes=int(not np.array_equal(pp["geometry"], pp["history"])))
                    for m in METHODS:
                        idx = a[f"top_{m}"][i]
                        idx = idx[idx >= 0]
                        if len(idx):
                            np.testing.assert_array_equal(legal[idx[0]], pp[m])
                        _, q, _ = prev.label_terms(d, pp[m][None], pp["additive"], y, v, cut)
                        counts.update({f"{m}_top5": int(strict[idx].any()), f"{m}_strict": int(q[0]), f"{m}_both": int(ok and q[0])})
                if active:
                    _, q, _ = prev.label_terms(d, pp[selected][None], pp["additive"], y, v, cut)
                    counts.update(release_views=1, release_both=int(ok and q[0]))
                pair = {f"{x}_vs_{z}": av.frontier.pair_reduction(pp[x][None], pp[z], y, v)[0].tolist()
                        for x, z in (("history", "geometry"), ("history", "additive"), ("history", "uniform"),
                                     ("geometry", "additive"), ("history", "prior_risk"))}
                rows.append(dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), metrics=metrics, pair=pair))
        beat("score", fold=k, views=len(rows), total=len(truth)*len(eq.SEEDS))
    metrics = {}
    for m in ARMS:
        risk = [r["metrics"][m]["risk"] for r in rows if r["metrics"][m]["risk"]["selected"]]
        metrics[m] = dict(edits=sum(x["selected"] for x in risk), edited=len(risk), risk_pass=sum(x["supported"] for x in risk),
            dual_risk_pass=sum(eq.ceiling.eligible_additive(r["metrics"][m]) for r in rows),
            easy_undefined=sum(x["completion_easy_risk_upper"] is None for x in risk),
            easy_over_2=sum((x["known_easy_net_degradation"] or 0.) > .02 for x in risk))
        metrics[m].update({q: av.interval(rows, lambda r: r["metrics"][m][q]) for q in ("ade", "fde", "gain", "hard", "tail")})
    for new, old_name in (("prior_risk", "risk_only"), ("prior_interaction", "interaction"), ("additive", "additive")):
        if metrics[new] != prior["metrics"][old_name]:
            raise ValueError("Frozen decision baseline changed")
    if len({metrics[m]["edits"] for m in ("additive", "uniform", *METHODS)}) != 1:
        raise ValueError("Matched edit budgets diverged")
    if counts["paired_ceiling"] != prior["counts"]["paired_ceiling"]:
        raise ValueError("Reference-relative oracle did not replay")
    terms = {k: {g: av.interval([r for r in rows if g == "all" or r["complete"]], lambda r: r["pair"][k])
                 for g in ("all", "complete")} for k in rows[0]["pair"]}
    cal = {m: {k: sum(z["calibration"][m][k] for z in frozen["folds"].values())
               for k in ("tested", "feasible", "local_strict", "strict_with_add", "accepted")} for m in METHODS}
    result = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", new_fits=2*len(frozen["folds"]),
        held_scenarios=len(truth), metrics=metrics, contrasts=terms, counts=dict(counts), calibration=cal,
        selected=dict(Counter(z["selected"] or "cv" for z in frozen["folds"].values())),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(LOCAL/"rows.json", rows)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps({k: result[k] for k in ("new_fits", "counts", "calibration", "selected")}, indent=2))


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
        key, data, source, roles, inner, outer, prior, summary = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, outer, prior)
        else:
            score(key, data, source, outer, prior, summary)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
