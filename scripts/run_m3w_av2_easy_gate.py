"""Test a two-part easy gate on fixed city-excluded AV2 choices."""

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
from sklearn.metrics import average_precision_score
from scripts import run_m3w_av2_selected_veto as veto
from src.world_model import m3w_av2_easy_gate as model

upper, prior, np, torch, av, eq, prev, study, base, old = (
    veto.upper, veto.prior, veto.np, veto.torch, veto.av, veto.eq, veto.prev, veto.study, veto.base, veto.old)
OUT, LOCAL = veto.OUT.parent/"av2_easy_gate_v1", veto.LOCAL.parent/"av2_easy_gate_v1"
MODES, PRIMARY = ("prior", "support", "direct", "factored", "conditional_tail"), "factored"
STEPS, SAVE, BATCH = 320, 40, 64


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, data, src, roles, inner, outer, margin, parent, _ = veto.register()
    p = veto.OUT/"summary.json"
    report = study.read(p)
    seal = study.read(study.checked(report["freeze"]))
    if (seal["registration"] != key or report["registration"] != key or seal["held_labels_used"]
            or report["reserved_outcomes_opened"] or not report["same_forced_paths_and_budget"]):
        raise ValueError("Require the completed out-of-city veto")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(v)) for v in (__file__, model.__file__)],
        parent=study.artifact(p), parent_freeze=report["freeze"], primary=PRIMARY, modes=list(MODES),
        targets="edited-easy reference exists; exact positive/signed easy passes conditional on support; "
                "two harm excess quantiles trained only where support exists",
        comparisons="prior independent selected veto; support-only reweight; direct easy joint event; "
                    "factored event; support times conditional upper-tail score",
        fixed="city-excluded training choices and outer upper choices, gain and overall harm; same exact calibration and 2%",
        steps=STEPS, checkpoint=SAVE, batch=BATCH, seeds=eq.SEEDS, torch_threads=4, num_workers=0,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, src, roles, inner, outer, margin, parent, seal, report


def easy_events(risk):
    support = risk["completion_easy_risk_upper"] is not None
    signed = risk["known_easy_net_degradation"]
    safe = support and signed is not None and risk["completion_easy_risk_upper"] <= .02 and signed <= .02
    return support, safe


def inputs(d, m, i, cut):
    x, ctx, opts, _ = prior.features(d, m, cut)
    return x[opts[i]], torch.cat((ctx.mean(0), ctx.amax(0)))


def target(d, m, i, p, y, v, cut):
    measure = prev.metrics(p, d, y, v, cut)
    s, safe = easy_events(measure["risk"])
    easy = upper.audit.actual(d, m, y, v, cut, i)[1:]
    return torch.tensor([s, safe, *easy], dtype=torch.float32), measure


def load_ranker(receipt, fold, city, width):
    identity = receipt["identity"]
    expect = dict(fold["identity"], donors=veto.ranker_donors(fold["identity"], city), recipient=city,
                  cut=identity["cut"], candidate_checkpoint=identity["candidate_checkpoint"])
    ck = torch.load(study.checked(receipt["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != expect or identity != expect or ck["step"] != veto.STEPS:
        raise ValueError("Incorrect nested ranker")
    nets = torch.nn.ModuleDict(dict(chooser=prior.model.Assignment(width), harm=upper.model.Harm(width)))
    nets.load_state_dict(ck["net"])
    return nets.eval()


def donor_rows(data, labels, role, inner, fold, seed, cut):
    raw = old.donor_forecasts(inner, data, fold["identity"]["held"], role, seed)
    rows = []
    for city, receipt in fold["nested"].items():
        models, count = None, 0
        local_cut = receipt["identity"]["cut"]
        for n, (y, v) in labels.items():
            if data[n]["city"] != city:
                continue
            d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
            m = old.action.menu(d, seed, local_cut)
            if m is None:
                continue
            if models is None:
                models = load_ranker(receipt, fold, city, m["x"].shape[1])
            paths, _, ix, _ = upper.rank(models["harm"], models["chooser"], d, m, local_cut)
            outer_menu = old.action.menu(d, seed, cut)
            for k in ("opts", "row", "col"):
                np.testing.assert_array_equal(m[k], outer_menu[k])
            j = ix["upper"]
            y_target, _ = target(d, outer_menu, j, paths["upper"], y, v, cut)
            rows.append((*inputs(d, outer_menu, j, cut), y_target))
            count += 1
        if count != receipt["selected_queries"]:
            raise ValueError("The previous donor query roster changed")
    if len(rows) != fold["fit"]["donor_queries"]:
        raise ValueError("Prior donor roster count did not replay")
    return tuple(torch.stack(v) for v in zip(*rows, strict=True))


def fit(ident, name, data):
    x, ctx, y = data
    torch.manual_seed(ident["seed"])
    net = model.EasyGate(x.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(ident["seed"]+17501)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incompatible checkpoint")
        start, trace = ck["step"], ck["trace"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        ix = rng.integers(len(x), size=BATCH)
        loss = model.loss(net(x[ix], ctx[ix]), y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=net.state_dict(),
                optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(path), trace=trace, training_queries=len(x),
                            easy_support=int(y[:, 0].sum()), easy_pass=int(y[:, 1].sum()))


def load_veto(fold):
    ck = torch.load(study.checked(fold["fit"]["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != fold["identity"] or ck["step"] != veto.STEPS:
        raise ValueError("Incorrect frozen veto")
    state = {k[len("selected."):]: v for k, v in ck["net"].items() if k.startswith("selected.")}
    net = upper.model.Harm(state["point.0.weight"].shape[1]).eval()
    net.load_state_dict(state)
    return net


@torch.inference_mode()
def rank(net, controls, d, m, cut):
    action, self_harm, old_veto = controls
    path, i, scores, harm = veto.rank({"selected": old_veto}, self_harm, action, d, m, cut)
    x, ctx, opts, row = prior.features(d, m, cut)
    a, b = action(x, ctx)
    gain = prior.model.margins(a, b, opts[i:i+1], row, torch.from_numpy(m["unit"]).float())[0, :2]
    temperature = prior.model.TEMPERATURE
    common = (torch.cat((gain, gain.new_tensor([-harm["selected"][0]])))/temperature).sigmoid().prod()
    edits, scene = inputs(d, m, i, cut)
    logits = net(edits[None], scene[None])[0]
    s, direct, factored, _ = model.probabilities(logits)
    scores = dict(prior=scores["selected"], support=float(scores["selected"]*s), direct=float(common*direct),
        factored=float(common*factored), conditional_tail=float(common*s*(-logits[3:]/temperature).sigmoid().prod()))
    return path, i, scores, model.probabilities(logits).numpy(), logits[3:].numpy()


def calibrate(net, controls, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {m: [] for m in MODES}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (m := old.action.menu(d, seed, cut)) is None:
            continue
        p, _, scores, _, _ = rank(net, controls, d, m, cut)
        ok = eq.ceiling.eligible_additive(prev.metrics(p, d, *cy[n], cut))
        for k, score in scores.items():
            rows[k].append(dict(score=score, strict=ok, add_ok=True))
    return {k: old.action.prior.calibration(rows[k]) for k in MODES}


def freeze(key, data, src, roles, inner, outer, margin, parent, old_veto):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            fold, q, producer = old_veto["folds"][name], parent["folds"][name], outer["folds"][name]
            cut = fold["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if fold["identity"] != dict(ident, registration=old_veto["registration"]):
                raise ValueError("Fold roles do not match parent")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                net, fit_receipt = fit(ident, name, donor_rows(data, dy, role, inner, fold, seed, cut))
                controls = upper.audit.load(margin["folds"][name]), veto.load_harm(q), load_veto(fold)
                table = calibrate(net, controls, old.load_outer(producer), data, cy, seed, cut)
                if table["prior"] != fold["calibration"]["selected"]:
                    raise ValueError("Prior calibration did not replay")
                with (np.load(study.checked(producer["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(q["forecast"]), allow_pickle=False) as chosen,
                      np.load(study.checked(fold["forecast"]), allow_pickle=False) as fixed):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], chosen[k])
                        np.testing.assert_array_equal(raw[k], fixed[k])
                    scores, prob, tail = {k: [] for k in MODES}, [], []
                    for i, (n, lo, hi) in enumerate(zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True)):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        if (m := old.action.menu(d, seed, cut)) is None:
                            s, prob_i, tail_i = dict.fromkeys(MODES, -1.), np.zeros(4), np.zeros(2)
                        else:
                            p, j, s, prob_i, tail_i = rank(net, controls, d, m, cut)
                            np.testing.assert_array_equal(p, chosen["upper"][lo:hi])
                            np.testing.assert_array_equal(j, chosen["ix_upper"][i])
                        np.testing.assert_array_equal(s["prior"], fixed["score_selected"][i])
                        for k in MODES:
                            scores[k].append(s[k])
                        prob.append(prob_i)
                        tail.append(tail_i)
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, prob=np.asarray(prob), tail=np.asarray(tail),
                        **{f"score_{k}": np.asarray(v) for k, v in scores.items()}))
                base.immutable(receipt, dict(identity=ident, fit=fit_receipt, forecast=bank, calibration=table))
            r = study.read(receipt)
            if r["identity"] != ident:
                raise ValueError("Incorrect checkpoint identity")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(old_veto["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, src, outer, parent, previous, report):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(previous["folds"]):
        raise ValueError("Require frozen easy scores")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows, gains, total, by_city = [], [], Counter(), {}
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        city = by_city.setdefault(held, Counter())
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(parent["folds"][name]["forecast"]), allow_pickle=False) as path,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(previous["folds"][name]["forecast"]), allow_pickle=False) as fixed):
            for k in ("names", "offsets"):
                for src_bank in (path, raw, fixed):
                    np.testing.assert_array_equal(z[k], src_bank[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Invalid held lineage")
            np.testing.assert_array_equal(z["score_prior"], fixed["score_selected"])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                s = dict(clip=str(n), city=held, seed=seed)
                g = dict.fromkeys(MODES, 0.)
                j = int(path["ix_upper"][i])
                if j >= 0:
                    d, (y, v) = data[n], truth[n]
                    m = old.action.menu(dict(d, bank=eq.expanded(d, raw["raw_eq"][lo:hi])), seed, cut)
                    chosen = path["upper"][lo:hi]
                    target_i, metrics = target(d, m, j, chosen, y, v, cut)
                    risk = metrics["risk"]
                    support, safe = target_i[:2].bool().tolist()
                    ok = eq.ceiling.eligible_additive(metrics)
                    stat = Counter(two_slots=1, easy_support=int(support), easy_pass=int(safe), eligible=int(ok))
                    rows.append(dict(s, y=target_i.tolist(), prob=z["prob"][i].tolist(),
                        coverage=(target_i[2:].numpy() <= z["tail"][i]).tolist(),
                        prior_coverage=(target_i[2:].numpy() <= fixed["harm_selected"][i, 1:]).tolist()))
                    for k in MODES:
                        t = f["calibration"][k]["threshold"]
                        active = t is not None and z[f"score_{k}"][i] >= t
                        stat.update({f"{k}_released": int(active), f"{k}_eligible": int(active and ok),
                            f"{k}_risk_fail": int(active and not risk["supported"]),
                            f"{k}_no_easy_support": int(active and not support),
                            f"{k}_easy_fail_with_support": int(active and support and not safe)})
                        if active:
                            delta = prior.cost.audit.direct_trajectory(chosen, d["cv"], d["cv"], y, v)
                            g[k] = None if delta is None else delta[2]
                    total.update(stat)
                    city.update(stat)
                gains.append(dict(s, gain=g))
        beat("score", fold=name, views=len(gains), total=len(truth)*len(eq.SEEDS))
    if (total["two_slots"] != report["counts"]["two_slots"] or total["eligible"] != report["counts"]["eligible"]
            or total["prior_released"] != report["counts"]["selected_released"]
            or total["prior_risk_fail"] != report["counts"]["selected_risk_fail"]):
        raise ValueError("Frozen eligibility or release changed")
    y, q = (np.asarray([r[k] for r in rows]) for k in ("y", "prob"))
    has_support = y[:, 0].astype(bool)
    truth_events = (y[:, 0], y[:, 1], y[:, 1], y[has_support, 1])
    event_metrics = {}
    for i, m in enumerate(("support", "direct", "factored", "conditional")):
        score_m = q[:, i] if i < 3 else q[has_support, i]
        t = truth_events[i]
        event_metrics[m] = dict(brier=float(np.mean((score_m-t)**2)), prevalence=float(t.mean()),
            ap=float(average_precision_score(t, score_m)), rows=len(t))
    delta = av.interval(rows, lambda r: (r["prob"][2]-r["y"][1])**2-(r["prob"][1]-r["y"][1])**2)
    supported = [r for r in rows if r["y"][0]]
    coverage = {k: np.mean([r[k] for r in supported], axis=0).tolist() for k in ("prior_coverage", "coverage")}
    cal = {k: {m: sum(f["calibration"][k][m] for f in seal["folds"].values())
        for m in ("tested", "feasible", "accepted", "local_strict")} for k in MODES}
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", primary=PRIMARY,
        new_fits=len(seal["folds"]), reused_inner_rankers=72, held_scenarios=len(truth),
        counts=dict(total), by_city={k: dict(v) for k, v in by_city.items()}, calibration=cal,
        event_metrics=event_metrics, paired_brier_factored_minus_direct=delta, conditional_coverage=coverage,
        released_gain_vs_cv={k: av.interval(gains, lambda r: r["gain"][k]) for k in MODES},
        same_forced_paths_and_budget=True, original_2pct=True, deployment_changed=False,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "event_metrics",
        "paired_brier_factored_minus_direct", "conditional_coverage")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, inner, outer, margin, parent, previous, report = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, margin, parent, previous)
        else:
            score(key, data, src, outer, parent, previous, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
