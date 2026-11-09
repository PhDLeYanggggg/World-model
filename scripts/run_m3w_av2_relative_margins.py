"""Compare dense relative margins with binary scene qualification on AV2 development."""

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
from scripts import run_m3w_av2_paired_qualification as pair
from src.evaluation.m3w_cv_anchor_budget import risk_excess
from src.world_model import m3w_av2_relative_margins as model

np, torch, av, eq, prev, study, base, old, cost = (
    pair.np, pair.torch, pair.av, pair.eq, pair.prev, pair.study, pair.base, pair.old, pair.cost)
OUT, LOCAL = pair.OUT.parent/"av2_relative_margins_v2", pair.LOCAL.parent/"av2_relative_margins_v2"
MODES, PRIMARY = ("binary", "dense"), "dense"
STEPS, SAVE, BATCH, OPTIONS = 320, 40, 24, 8


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), time=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent_key, data, src, roles, inner, outer, control, references, _, audit = pair.register()
    p = pair.OUT/"summary.json"
    r = study.read(p)
    seal = study.read(study.checked(r["freeze"]))
    if (r["registration"] != parent_key or seal["registration"] != parent_key or seal["held_labels_used"]
            or r["reserved_outcomes_opened"] or set(seal["folds"]) != set(references["folds"])):
        raise ValueError("Require the sealed paired development evaluation")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(p), parent_freeze=r["freeze"], primary=PRIMARY, modes=list(MODES),
        inputs="same past geometry as paired qualification, fixed additive reference plus option-reference feature differences",
        labels="matched event, additive eligibility, candidate edited-easy support, and dense margins: relative pair and trajectory, "
               "minimum CV/OLS gain, minimum of the three unchanged risk margins (masked without edited easy support)",
        correction="legacy vectorized all-risk yielded NaN for a wholly missing actor; use zero known error and debit edited unknown actors, "
                   "verify recovered labels against scalar risk checks and keep the legacy oracle as a separate replay",
        matching="same EqMotion forecast bank, exact two edit magnitudes, fixed independent additive arm",
        draws="same scenes/options, initialization, optimizer and steps; one positive and one near-miss per query when present",
        calibration="original exact both-arm qualification at every nonempty prefix, CV fallback for both arms",
        bootstrap="3000 paired city draws after averaging seeds within scenario",
        steps=STEPS, checkpoint_every=SAVE, batch=BATCH, options_per_query=OPTIONS,
        temperature=model.TEMPERATURE, seeds=eq.SEEDS, torch_threads=4, num_workers=0,
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, src, roles, inner, outer, control, references, audit, seal, r


def targets(d, prepared, y, valid, cut):
    p, a, *_, unit = prepared
    c, o = d["cv"], d["ols4"]
    safe = np.where(valid[..., None], y, 0.)
    err = np.linalg.norm(p-safe, axis=-1)
    norm = lambda x: np.linalg.norm(x, axis=-1)
    error = np.where(valid, norm(c-safe), 0.)
    mass = float(error.sum())
    mean = error.sum(1)/valid.sum(1).clip(1)
    easy = (mean > 0.) & (mean <= cut[0])
    cap = .02*mean[easy].sum()
    radius = av.motion.peak(p-c)
    s = ((radius[:, easy] > 0.).any(1)) & (mass > 0.)
    excess = risk_excess(c, p.swapaxes(0, 1), safe, valid, cut[0]).sum(0)
    signed = np.where(valid, err-norm(c-safe), 0.).sum(2)/valid.sum(1).clip(1)
    risk_margin = np.minimum(-excess[:, 0], np.minimum(-excess[:, 1], cap-signed[:, easy].sum(1)))
    total = lambda ref: np.where(valid, norm(ref-safe)-err, -norm(p-ref)).sum((1, 2))
    trajectory, floor = total(a), np.minimum(total(c), total(o))
    relative_pair = av.frontier.pair_reduction(p, a, safe, valid)[:, 1]
    eligible = eq.ceiling.eligible_additive(prev.metrics(a, d, y, valid, cut))
    relative = (relative_pair > 1e-10) & (trajectory >= -1e-10*mass)
    qualified = s & (floor > 0.) & (risk_margin >= 0.) & relative
    scale = valid.size*unit
    return np.column_stack((qualified & eligible, np.full(len(p), eligible), s,
        relative_pair/unit, trajectory/scale, floor/scale, risk_margin/(len(c)*unit))).astype(np.float32)


def verify_label(d, p, a, y, valid, cut, anchor_ok):
    eligible = eq.ceiling.eligible_additive(prev.metrics(p, d, y, valid, cut))
    traj = cost.audit.direct_trajectory(p, a, d["cv"], y, valid)
    rel = av.frontier.pair_reduction(p[None], a, y, valid)[0, 1]
    return bool(eligible and anchor_ok and traj is not None and traj[2] >= -1e-8 and rel > 1e-10)


def audit_labels(d, s, y, valid, cut, t, check=()):
    p, a = s[:2]
    legacy, anchor = pair.truth_terms(d, p, a, y, valid, cut)
    matched = t[:, 0].astype(bool)
    legacy &= anchor
    different = np.flatnonzero(matched != legacy)
    if different.size and valid.any(1).all():
        raise ValueError("Corrected labels disagree in a fully supported actor roster")
    for j in np.union1d(different, check).astype(int):
        if matched[j] != verify_label(d, p[j], a, y, valid, cut, anchor):
            raise ValueError("Vector labels must match the scalar evaluator")
    return matched, legacy, anchor, different


def examples(inner, data, truth, role, ident, refs):
    held, seed, cut = (ident[k] for k in ("held", "seed", "cut"))
    raw = old.donor_forecasts(inner, data, held, role, seed)
    anchors = {city: pair.inner_action(ref, refs["identity"], city) for city, ref in refs["references"].items()}
    if set(raw) != set(truth):
        raise ValueError("Donor forecast and label rosters disagree")
    out = []
    for n, (y, v) in truth.items():
        d = dict(data[n], bank=eq.expanded(data[n], raw[n]))
        net, ac = anchors[d["city"]]
        if (s := cost.slate(d, seed, cut, net, action_cut=ac)) is not None:
            x, a, idx = pair.features(s)
            t = targets(d, s, y, v, cut)[idx]
            if not np.isfinite(t).all():
                raise ValueError("Nonfinite donor margins")
            positives = np.flatnonzero(t[:, 0])
            near = np.flatnonzero((t[:, 0] == 0) & (t[:, 1] > 0) & (t[:, 2] > 0))
            out.append((x[idx], a, torch.from_numpy(t), positives, near))
    return out


def draw(data, rng):
    batches = [[], [], []]
    for i in rng.integers(len(data), size=BATCH):
        x, a, t, positives, near = data[i]
        j = rng.integers(len(x), size=OPTIONS)
        for k, group in enumerate((positives, near)):
            if len(group):
                j[k] = rng.choice(group)
        for b, v in zip(batches, (x[j], a.expand_as(x[j]), t[j]), strict=True):
            b.append(v)
    return [torch.cat(v) for v in batches]


def fit(ident, name, data):
    if not data:
        raise ValueError("Empty relative margin training set")
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+35387)
    nets = torch.nn.ModuleDict({m: model.RelativeMargins(data[0][0].shape[1]) for m in MODES})
    nets["dense"].load_state_dict(nets["binary"].state_dict())
    optimizer = torch.optim.AdamW(nets.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Incompatible checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        optimizer.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
    for step in range(start+1, STEPS+1):
        x, a, t = draw(data, rng)
        losses = [model.loss(nets[k](x, a), t, k == PRIMARY) for k in MODES]
        optimizer.zero_grad(set_to_none=True)
        sum(losses).backward()
        for net in nets.values():
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % SAVE == 0:
            trace.append([float(x.detach()) for x in losses])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=nets.state_dict(),
                            optimizer=optimizer.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(path), trace=trace, queries=len(data),
                            positive_queries=sum(len(r[3]) > 0 for r in data),
                            positive_options=sum(len(r[3]) for r in data))


@torch.inference_mode()
def rank(nets, prepared):
    x, a, ix = pair.features(prepared)
    indices, confidence = {}, {}
    for k in MODES:
        s = model.scores(nets[k](x[ix], a), k == PRIMARY).numpy()
        top = ix[np.argsort(-s, kind="stable")[:5]]
        indices[k] = np.pad(top, (0, 5-len(top)), constant_values=-1)
        confidence[k] = float(np.exp(s.max()))
    return indices, confidence


def calibrate(nets, action, forecaster, data, truth, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(truth))
    tally, rows = Counter(), {k: [] for k in MODES}
    for n, lo, hi in zip(truth, offsets[:-1], offsets[1:], strict=True):
        d = dict(data[n], bank=eq.expanded(data[n], raw[lo:hi]))
        if (s := cost.slate(d, seed, cut, action)) is None:
            continue
        idx, scores = rank(nets, s)
        t = targets(d, s, *truth[n], cut)
        match, legacy, add, different = audit_labels(
            d, s, *truth[n], cut, t, np.concatenate([idx[k][idx[k] >= 0] for k in MODES]))
        tally.update(two_slots=1, paired_available=int(match.any()), legacy_paired_available=int(legacy.any()),
                     corrected_candidates=len(different), corrected_queries=int(bool(len(different))))
        for k in MODES:
            top = idx[k][idx[k] >= 0]
            tally.update({f"{k}_top1": int(match[top[0]]), f"{k}_top5": int(match[top].any())})
            rows[k].append(dict(score=scores[k], strict=bool(match[top[0]]), add_ok=add))
    return dict(counts=dict(tally), modes={k: old.action.prior.calibration(rows[k]) for k in MODES})


def freeze(key, data, src, roles, inner, outer, control, references, audit, previous):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(src, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            ref, producer = references["folds"][name], outer["folds"][name]
            cut = ref["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(ident[k] != ref["identity"][k] or ident[k] != previous["folds"][name]["identity"][k]
                   for k in ("held", "calibration", "donors", "seed", "cut")):
                raise ValueError("Fold lineage mismatch")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                nets, fitted = fit(ident, name, examples(inner, data, dy, role, ident, ref))
                action = pair.prior.load_action(control["folds"][name])
                table = calibrate(nets, action, old.load_outer(producer), data, cy, seed, cut)
                for k, legacy in (("two_slots", "two_slots"), ("legacy_paired_available", "paired_available")):
                    if table["counts"][k] != audit["by_fold"][name][legacy]:
                        raise ValueError("Changed calibration opportunity set")
                with (np.load(study.checked(producer["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as fixed):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], fixed[k])
                    tops, scores = {k: [] for k in MODES}, {k: [] for k in MODES}
                    for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                        d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                        s = cost.slate(d, seed, cut, action)
                        if s is None:
                            idx, conf = dict.fromkeys(MODES, np.full(5, -1)), dict.fromkeys(MODES, -1.)
                        else:
                            np.testing.assert_array_equal(s[1], fixed["additive"][lo:hi])
                            idx, conf = rank(nets, s)
                        for k in MODES:
                            tops[k].append(idx[k])
                            scores[k].append(conf[k])
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(names=raw["names"],
                        offsets=raw["offsets"], registration=key, **{f"top_{k}": np.asarray(v) for k, v in tops.items()},
                        **{f"score_{k}": np.asarray(v) for k, v in scores.items()}))
                base.immutable(receipt, dict(identity=ident, fit=fitted, forecast=bank, calibration=table))
            r = study.read(receipt)
            if r["identity"] != ident:
                raise ValueError("Incompatible fold receipt")
            study.checked(r["fit"]["checkpoint"])
            study.checked(r["forecast"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(previous["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, src, outer, control, previous, report):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(previous["folds"]):
        raise ValueError("Require sealed heads, scores and calibration")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    total, by_city, rows, releases = Counter(), {}, [], {k: [] for k in MODES}
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        city = by_city.setdefault(held, Counter())
        action = pair.prior.load_action(control["folds"][name])
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a,
              np.load(study.checked(previous["folds"][name]["forecast"]), allow_pickle=False) as old_score,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                for bank in (a, old_score, raw):
                    np.testing.assert_array_equal(z[k], bank[k])
            if z["registration"].item() != key or any(data[n]["city"] != held for n in z["names"]):
                raise ValueError("Invalid held readout")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi])), truth[n]
                row = dict(clip=str(n), city=held, seed=seed)
                s = cost.slate(d, seed, cut, action)
                stats = {}
                for k in MODES:
                    row[f"{k}_released_gain"] = [0., 0.]
                if s is not None:
                    paths, anchor = s[:2]
                    np.testing.assert_array_equal(anchor, a["additive"][lo:hi])
                    t = targets(d, s, y, v, cut)
                    ref = old_score["top_direct"][i]
                    check = [ref[0], *(z[f"top_{k}"][i, 0] for k in MODES)]
                    label, legacy, add, different = audit_labels(d, s, y, v, cut, t, check)
                    stats.update(two_slots=1, anchor_eligible=int(add), paired_available=int(label.any()),
                                 legacy_paired_available=int(legacy.any()), corrected_candidates=len(different),
                                 corrected_queries=int(bool(len(different))))
                    stats.update(prior_top1=int(label[ref[0]]), legacy_prior_top1=int(legacy[ref[0]]))
                    for k in MODES:
                        ix = z[f"top_{k}"][i]
                        ix = ix[ix >= 0]
                        j = ix[0]
                        threshold = fold["calibration"]["modes"][k]["threshold"]
                        active = threshold is not None and z[f"score_{k}"][i] >= threshold
                        stats.update({f"{k}_top1": int(label[j]), f"{k}_top5": int(label[ix].any()),
                                      f"{k}_released": int(active), f"{k}_released_match": int(active and label[j])})
                        if active:
                            traj = cost.audit.direct_trajectory(paths[j], anchor, d["cv"], y, v)
                            rel = float(av.frontier.pair_reduction(paths[j:j+1], anchor, y, v)[0, 1])
                            m = prev.metrics(paths[j], d, y, v, cut)
                            releases[k].append(dict(fold=name, matched_pass=bool(label[j]),
                                anchor_eligible=add, risk=m["risk"], pair_gain_lower=rel,
                                trajectory_gain_lower_pp=traj[2] if traj else None))
                            if traj is None:
                                raise ValueError("Released scene has no valid reference error mass")
                            row[f"{k}_released_gain"] = [traj[2], rel]
                total.update(stats)
                city.update(stats)
                rows.append(dict(row, **stats))
    expected = report["counts"]
    for k, ref in (("two_slots", "two_slots"), ("anchor_eligible", "additive_eligible"),
                   ("legacy_paired_available", "paired_ceiling"), ("legacy_prior_top1", "direct_both")):
        if total[k] != expected[ref]:
            raise ValueError(f"Changed held opportunity set: {k}")
    paired = [r for r in rows if r.get("paired_available")]
    fields = ("two_slots", "anchor_eligible", "paired_available", "prior_top1", "legacy_paired_available",
              "legacy_prior_top1", "corrected_candidates", "corrected_queries",
              *(f"{k}_{m}" for k in MODES for m in ("top1", "top5", "released", "released_match")))
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run",
        counts={k: total[k] for k in fields}, by_city={c: {k: n[k] for k in fields} for c, n in by_city.items()},
        calibration={k: sum(f["calibration"]["modes"][k]["accepted"] for f in seal["folds"].values()) for k in MODES},
        top1_dense_minus_binary_on_oracle_supported=av.interval(paired, lambda r: r["dense_top1"]-r["binary_top1"]),
        matched_release_vs_fixed={k: av.interval(rows, lambda r: r[f"{k}_released_gain"]) for k in MODES},
        released_gain_order=["trajectory_lower_pp", "pair_lower_native"], releases=releases,
        original_2pct=True, reserved_outcomes_opened=False, independent_confirmation=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration", "top1_dense_minus_binary_on_oracle_supported",
                                                "matched_release_vs_fixed")}, indent=2))


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
        key, data, src, roles, inner, outer, control, references, audit, previous, report = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, references, audit, previous)
        else:
            score(key, data, src, outer, control, previous, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
