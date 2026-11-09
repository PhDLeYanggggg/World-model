"""Test calibration scores without changing any past-replay substitution."""

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
from scripts import run_m3w_av2_incumbent_timing as timing
from src.world_model import m3w_av2_replay_hurdle as model

np, torch, av, eq, prev, study, base, old, cost = (
    timing.np, timing.torch, timing.av, timing.eq, timing.prev, timing.study, timing.base, timing.old, timing.cost)
OUT, LOCAL = timing.OUT.parent/"av2_replay_hurdle_v1", timing.LOCAL.parent/"av2_replay_hurdle_v1"
MODES = ("replay", "anchor_replay", "direct", "factored")
STEPS, SAVE, BATCH = 320, 40, 96


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    key, data, source, roles, inner, outer, control, refs, _ = timing.register()
    p = timing.OUT/"summary.json"
    r = study.read(p)
    frozen = study.read(study.checked(r["freeze"]))
    if (key != frozen["registration"] or key != r["registration"] or frozen["held_labels_used"]
            or r["reserved_outcomes_opened"] or set(frozen["folds"]) != set(refs["folds"])):
        raise ValueError("Require the completed and frozen timing diagnostic")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(p), parent_freeze=r["freeze"], modes=MODES, primary="factored",
        paths="fixed incumbent and positive-score retrospective replay's single replacement; no option reranking",
        supervision="A = incumbent exact eligibility; J = matched qualification; J given A; "
                    "conditional loss masked outside A, unweighted binary log losses",
        training="donor replay decisions constructed with city-excluded additive and forecast checkpoints; "
                 "outer calibration and held never used to fit or normalize",
        release="every distinct nonempty calibration prefix under the original exact both-arm checks; "
                "both forecast arms use CV outside the prefix",
        steps=STEPS, checkpoint=SAVE, batch=BATCH, threads=4, num_workers=0, seeds=eq.SEEDS,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, inner, outer, control, refs, frozen, r


def proposal(d, a, cut):
    if a is None or (s := timing.prepare(d, a, cut)) is None:
        return 0, 0., None, None
    raw = timing.local.replay_score(s[3])
    j = int(raw.argmax())
    if j == 0:
        return j, float(raw[j]), s, None
    paths, a_x = timing.inputs(s, True)
    return j, float(raw[j]), s, (a_x, paths[j-1], a_x.new_tensor(raw[j]))


def label(d, a, p, y, v, cut):
    anchor = eq.ceiling.eligible_additive(prev.metrics(a, d, y, v, cut))
    matched = timing.rel.verify_label(d, p, a, y, v, cut, anchor)
    return anchor, matched


def examples(data, dy, role, inner, ref, seed, cut):
    raw = old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(dy):
        raise ValueError("Donor inputs and labels disagree")
    heads = {c: timing.rel.pair.inner_action(z, ref["identity"], c) for c, z in ref["references"].items()}
    rows = []
    for n, (y, v) in dy.items():
        d = data[n]
        net, ac = heads[d["city"]]
        a = timing.incumbent(d, seed, cut, raw[n], net, action_cut=ac)
        j, _, s, x = proposal(d, a, cut)
        if x is not None:
            rows.append((*x, torch.tensor(label(d, a, s[0][j], y, v, cut), dtype=torch.float32)))
    if not rows:
        raise ValueError("Empty donor replay proposals")
    return tuple(torch.stack(v) for v in zip(*rows, strict=True))


def fit(ident, name, samples):
    a, x, replay, y = samples
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+8341)
    net = model.ReplayHurdle(a.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong replay gate checkpoint")
        start, trace = ck["step"], ck["trace"]
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        j = rng.integers(len(a), size=BATCH)
        loss = model.loss(net(a[j], x[j], replay[j]), y[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=net.state_dict(),
                            optimizer=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), trace=trace, proposals=len(a),
                            anchor_eligible=int(y[:, 0].sum()), matched=int(y[:, 1].sum()))


@torch.inference_mode()
def predict(net, x, replay):
    if x is None:
        return [replay, -1., -1., -1., -1., -1.]
    r = model.scores(net(*x), x[-1]).tolist()
    r[0] = replay
    return r


def calibrate(net, action, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows = {k: [] for k in MODES}
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = timing.incumbent(d, seed, cut, raw[lo:hi], action)
        j, value, s, x = proposal(d, a, cut)
        if x is not None:
            aa, joint = label(d, a, s[0][j], *cy[n], cut)
            for k, score in zip(MODES, predict(net, x, value)[:len(MODES)], strict=True):
                rows[k].append(dict(score=score, strict=joint, add_ok=aa))
    return {k: old.action.prior.calibration(v) for k, v in rows.items()}


def freeze(key, data, source, roles, inner, outer, control, refs, prior):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            orig, cf, of, ref = prior["folds"][name], control["folds"][name], outer["folds"][name], refs["folds"][name]
            cut = orig["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(ident[k] != item["identity"][k] for item in (orig, ref, cf)
                   for k in ("held", "calibration", "donors", "seed", "cut")):
                raise ValueError("Invalid replay gate roles")
            p = LOCAL/f"{name}.json"
            if not p.exists():
                net, fitted = fit(ident, name, examples(data, dy, role, inner, ref, seed, cut))
                action = timing.rel.pair.prior.load_action(cf)
                table = calibrate(net, action, old.load_outer(of), data, cy, seed, cut)
                if table["replay"] != orig["calibration"]["rules"]["replay"]:
                    raise ValueError("Replay calibration changed")
                with (np.load(study.checked(cf["forecast"]), allow_pickle=False) as a,
                      np.load(study.checked(orig["forecast"]), allow_pickle=False) as z):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(a[k], z[k])
                    scores = []
                    for i, (n, lo, hi) in enumerate(zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True)):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held city")
                        anchor = a["additive"][lo:hi] if len(eq.radii(d, seed)) == 2 else None
                        j, value, s, x = proposal(d, anchor, cut)
                        if j != z["choice_replay"][i] or (timing.digest(s) if s is not None else "") != z["digest"][i]:
                            raise ValueError("Past replay choices changed")
                        if s is not None and value != z["score_replay"][i]:
                            raise ValueError("Raw replay score changed")
                        scores.append(predict(net, x, value if s is not None else -1.))
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=a["names"],
                        offsets=a["offsets"], registration=key, scores=np.asarray(scores)))
                base.immutable(p, dict(identity=ident, calibration=table, fit=fitted, forecast=bank))
            f = study.read(p)
            if f["identity"] != ident:
                raise ValueError("Wrong sealed gate identity")
            study.checked(f["fit"]["checkpoint"])
            study.checked(f["forecast"])
            done[name] = f
            beat("freeze", fold=name, completed=len(done), total=len(prior["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def probability_quality(rows, event, value):
    y = np.asarray([r[event] for r in rows])
    p = np.asarray([r[value] for r in rows])
    return dict(count=len(y), prevalence=float(y.mean()), brier=float(((y-p)**2).mean()),
                ap=float(average_precision_score(y, p)) if y.any() else None)


def score(key, data, source, control, prior, parent):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(prior["folds"]):
        raise ValueError("Require all release decisions frozen")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    counts, by_city, rows, releases = Counter(), {}, [], {k: [] for k in MODES}
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        city = by_city.setdefault(held, Counter())
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(prior["folds"][name]["forecast"]), allow_pickle=False) as choice,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], a[k])
                np.testing.assert_array_equal(z[k], choice[k])
            if z["registration"].item() != key:
                raise ValueError("Wrong sealed score")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                if d["city"] != held:
                    raise ValueError("Wrong held source")
                if len(eq.radii(d, seed)) != 2:
                    continue
                anchor = a["additive"][lo:hi]
                j, _, s, x = proposal(d, anchor, cut)
                if j != choice["choice_replay"][i]:
                    raise ValueError("Changed frozen replay choice")
                row = dict(clip=str(n), city=held, seed=seed, proposal=x is not None)
                stat = Counter(two_slots=1)
                if x is not None:
                    candidate = s[0][j]
                    aa, joint = label(d, anchor, candidate, y, v, cut)
                    ps = z["scores"][i]
                    row.update(anchor=aa, joint=joint, direct=float(ps[2]), factored=float(ps[3]),
                               p_anchor=float(ps[4]), conditional=float(ps[5]))
                    stat.update(proposals=1, anchor_eligible=int(aa), matched=int(joint))
                for k in MODES:
                    row[f"{k}_gain"] = [0., 0.]
                if x is not None:
                    for k, s_i in zip(MODES, ps[:len(MODES)], strict=True):
                        t = f["calibration"][k]["threshold"]
                        active = t is not None and s_i >= t
                        stat.update({f"{k}_released": int(active), f"{k}_pass": int(active and joint)})
                        if active:
                            gain = cost.audit.direct_trajectory(candidate, anchor, d["cv"], y, v)
                            pair = av.frontier.pair_reduction(candidate[None], anchor, y, v)[0, 1]
                            met = prev.metrics(candidate, d, y, v, cut)
                            row[f"{k}_gain"] = [gain[2], float(pair)] if gain is not None else None
                            releases[k].append(dict(fold=name, match=joint, anchor_eligible=aa, risk=met["risk"],
                                                    gain=met["gain"], relative=row[f"{k}_gain"]))
                rows.append(row)
                counts.update(stat)
                city.update(stat)
        beat("score", fold=name, scored=len(rows))
    if (counts["two_slots"], counts["matched"], counts["replay_released"]) != (
            parent["counts"]["two_slots"], parent["counts"]["replay_top"], parent["counts"]["replay_release"]):
        raise ValueError("Frozen replay results changed")
    use = [r for r in rows if r["proposal"]]
    cond = [r for r in use if r["anchor"]]
    fields = ("two_slots", "proposals", "anchor_eligible", "matched", *(f"{k}_{t}" for k in MODES for t in ("released", "pass")))
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run", fits=len(seal["folds"]),
        counts={k: counts[k] for k in fields}, by_city={k: {m: v[m] for m in fields} for k, v in by_city.items()},
        probability=dict(anchor=probability_quality(use, "anchor", "p_anchor"),
            conditional=probability_quality(cond, "joint", "conditional"),
            **{k: probability_quality(use, "joint", k) for k in ("direct", "factored")}),
        paired_brier_factored_minus_direct=av.interval(use, lambda r: (r["joint"]-r["factored"])**2-(r["joint"]-r["direct"])**2),
        calibration={k: sum(v["calibration"][k]["accepted"] for v in seal["folds"].values()) for k in MODES},
        releases=releases, released_gain={k: av.interval(rows, lambda r: r[f"{k}_gain"]) for k in MODES},
        gain_order=["trajectory_lower_pp", "pair_lower_native"], identical_replay_paths=True, original_2pct=True,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "probability", "paired_brier_factored_minus_direct", "calibration")}, indent=2))


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
        key, data, source, roles, inner, outer, control, refs, prior, parent = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, outer, control, refs, prior)
        else:
            score(key, data, source, control, prior, parent)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
