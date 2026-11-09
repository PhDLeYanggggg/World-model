"""Test a local, incumbent-preserving timing replacement on sealed AV2 development."""

import argparse
from collections import Counter
import fcntl
import hashlib
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_relative_margins as rel
from scripts import run_m3w_av2_temporal_refit as refit
from src.world_model import m3w_av2_forecast_cost as geometry
from src.world_model import m3w_av2_incumbent_timing as local

np, torch, av, eq, prev, study, base, old, cost = (
    rel.np, rel.torch, rel.av, rel.eq, rel.prev, rel.study, rel.base, rel.old, rel.cost)
OUT, LOCAL = rel.OUT.parent/"av2_incumbent_timing_v1", rel.LOCAL.parent/"av2_incumbent_timing_v1"
MODES, PRIMARY = ("replay", "geometry", "history"), "history"
STEPS, SAVE, BATCH = 320, 40, 128


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=refit.tm.os.getpid(), epoch=refit.tm.time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    pkey, data, source, roles, inner, outer, control, refs, _, _, _ = rel.register()
    rpath = refit.OUT/"summary.json"
    ref = study.read(rpath)
    ref_seal = study.read(study.checked(ref["freeze"]))
    qpath = rel.OUT/"summary.json"
    parent = study.read(qpath)
    frozen = study.read(study.checked(parent["freeze"]))
    if (frozen["registration"] != pkey or frozen["held_labels_used"] or ref_seal["held_labels_used"]
            or parent["reserved_outcomes_opened"] or ref["reserved_outcomes_opened"]
            or set(ref_seal["folds"]) != set(refs["folds"])):
        raise ValueError("Require sealed incumbent and timing diagnostics")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(p)) for p in (__file__, local.__file__,
                        geometry.__file__, local.history.__file__, rel.model.__file__)],
        parent=study.artifact(rpath), previous_qualification=study.artifact(qpath),
        primary=PRIMARY, modes=list(MODES), original_2pct=True,
        menu="incumbent plus the 3-by-3 combinations of original, early and late timing at the incumbent's two recipients and peaks",
        comparator="fixed original independent path, same query/forecasters/edited actors; no reassignment or extra candidates",
        training="donor-only exact matched event and component margins; geometry-only and history replay share updates, "
                 "equal architecture and sampled options; exclude candidate copies of incumbent",
        features="relative path geometry, causal edit shapes and retrospective compatibility with the observed 8 steps",
        calibration="unchanged strict both-arm 2% and dual-floor checks on every nonempty prefix; "
                    "report incumbent-preserving substitution separately from original release with both arms CV outside prefix",
        steps=STEPS, checkpoint_every=SAVE, batch=BATCH, seeds=eq.SEEDS, torch_threads=4, num_workers=0,
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, roles, inner, outer, control, refs, parent


def prepare(d, a, cut):
    p, edited, profile = local.paths(d["cv"], a)
    if len(p) <= 1:
        return None
    geom, _, unit = geometry.describe(d["h"], d["cv"], d["ols4"], p, a, cut[0])
    probes = local.history.features(d["h"], d["cv"], p, edited, cut[0])
    mode = np.eye(3)[profile].sum(1)
    prefix = np.column_stack((geom, mode)).astype(np.float32)
    return p, unit, prefix, probes


def inputs(s, use_history):
    _, _, geo, past = s
    x = torch.from_numpy(np.column_stack((geo, past if use_history else np.zeros_like(past))))
    return x[1:], x[0]


@torch.inference_mode()
def incumbent(d, seed, cut, raw, action, action_cut=None):
    d = dict(d, bank=eq.expanded(d, raw))
    m = old.action.menu(d, seed, cut if action_cut is None else action_cut)
    if m is None:
        return None
    z = action(torch.from_numpy(m["x"])).numpy()
    _, ix, _ = old.action.rank(z, m["opts"], m["unit"][m["row"]])
    return av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"][ix[:1]])[0]


def targets(d, s, a, y, v, cut):
    p, unit, *_ = s
    x = rel.targets(d, (p, a, unit), y, v, cut)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite timing supervision")
    for j in range(len(p)):
        if bool(x[j, 0]) != rel.verify_label(d, p[j], a, y, v, cut, bool(x[j, 1])):
            raise ValueError("Local paired labels must replay exact scalar rules")
    return x


def examples(data, truth, role, inner, ref, seed, cut):
    raw = old.donor_forecasts(inner, data, ref["identity"]["held"], role, seed)
    if set(raw) != set(truth):
        raise ValueError("Missing nested forecast")
    models = {k: rel.pair.inner_action(v, ref["identity"], k) for k, v in ref["references"].items()}
    xx, aa, yy = [], [], []
    for n, (y, v) in truth.items():
        d = data[n]
        net, ac = models[d["city"]]
        a = incumbent(d, seed, cut, raw[n], net, action_cut=ac)
        if a is not None and (s := prepare(d, a, cut)) is not None:
            x, anchor = inputs(s, True)
            xx.append(x)
            aa.append(anchor.expand_as(x))
            yy.append(torch.from_numpy(targets(d, s, a, y, v, cut)[1:]))
    if not xx:
        raise ValueError("Empty donor timing roster")
    return torch.cat(xx), torch.cat(aa), torch.cat(yy), len(xx)


def fit(ident, name, data):
    x, a, y, queries = data
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+37013)
    nets = torch.nn.ModuleDict({k: rel.model.RelativeMargins(x.shape[-1]) for k in ("geometry", "history")})
    nets["history"].load_state_dict(nets["geometry"].state_dict())
    opt = torch.optim.AdamW(nets.parameters(), lr=.001)
    path = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if path.exists():
        ck = torch.load(path, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Invalid timing substitution checkpoint")
        start, trace = ck["step"], ck["trace"]
        nets.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
    positive = np.flatnonzero(y[:, 0].numpy())
    for step in range(start+1, STEPS+1):
        i = rng.integers(len(x), size=BATCH)
        if len(positive):
            i[:BATCH//4] = rng.choice(positive, size=BATCH//4)
        xa, ar = x[i], a[i]
        xg, ag = xa.clone(), ar.clone()
        xg[:, -local.history.WIDTH:] = 0.
        ag[:, -local.history.WIDTH:] = 0.
        loss = [rel.model.loss(nets[k](z, h), y[i], True)
                for k, z, h in (("geometry", xg, ag), ("history", xa, ar))]
        opt.zero_grad(set_to_none=True)
        sum(loss).backward()
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(z.detach()) for z in loss])
            tmp = path.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, trace=trace, net=nets.state_dict(),
                            opt=opt.state_dict(), rng=rng.bit_generator.state), tmp)
            tmp.replace(path)
            beat("fit", fold=name, step=step, total=STEPS)
    return nets.eval(), dict(checkpoint=study.artifact(path), trace=trace,
                            donor_queries=queries, donor_options=len(x), positive_options=len(positive))


@torch.inference_mode()
def rank(nets, s):
    replay = local.replay_score(s[3])
    j = int(np.argmax(replay))
    picks, scores = dict(replay=j), dict(replay=float(replay[j]))
    for k, net in nets.items():
        z = rel.model.scores(net(*inputs(s, k == "history")), True)
        j = int(z.argmax())
        picks[k], scores[k] = j+1, float(z[j].exp())
    return picks, scores


def calibrate(nets, action, forecaster, data, cy, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(cy))
    rows, tally = {k: [] for k in MODES}, Counter()
    for n, lo, hi in zip(cy, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or (s := prepare(d, a, cut)) is None:
            continue
        ix, score = rank(nets, s)
        t = targets(d, s, a, *cy[n], cut)
        tally.update(scenes=1, anchor_eligible=int(t[0, 1]), pair_available=int(t[:, 0].any()))
        for k in MODES:
            tally[f"{k}_matched_top"] += int(t[ix[k], 0])
            rows[k].append(dict(score=score[k], strict=bool(t[ix[k], 0]), add_ok=bool(t[0, 1])))
    return dict(counts=dict(tally), rules={k: old.action.prior.calibration(rows[k]) for k in MODES})


def digest(s):
    return hashlib.sha256(np.ascontiguousarray(s[0]).tobytes()).hexdigest()


def freeze(key, data, source, roles, inner, outer, control, refs):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        dy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] in donors])
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            cf, of, ref = control["folds"][name], outer["folds"][name], refs["folds"][name]
            cut = ref["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            for b in (cf, ref):
                if any(ident[k] != b["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut")):
                    raise ValueError("Wrong incumbent training role")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                nets, fitted = fit(ident, name, examples(data, dy, role, inner, ref, seed, cut))
                action = rel.pair.prior.load_action(cf)
                cal_table = calibrate(nets, action, old.load_outer(of), data, cy, seed, cut)
                with (np.load(study.checked(cf["forecast"]), allow_pickle=False) as a,
                      np.load(study.checked(of["forecast"]), allow_pickle=False) as raw):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(a[k], raw[k])
                    choices, scores, sha = {k: [] for k in MODES}, {k: [] for k in MODES}, []
                    for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held roster")
                        old_a = incumbent(d, seed, cut, raw["raw_eq"][lo:hi], action)
                        if old_a is not None:
                            np.testing.assert_array_equal(old_a, a["additive"][lo:hi])
                        s = prepare(d, old_a, cut) if old_a is not None else None
                        ix, sc = rank(nets, s) if s is not None else (dict.fromkeys(MODES, 0), dict.fromkeys(MODES, -1.))
                        for k in MODES:
                            choices[k].append(ix[k])
                            scores[k].append(sc[k])
                        sha.append(digest(s) if s is not None else "")
                    f = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(
                        names=a["names"], offsets=a["offsets"], registration=key, digest=np.asarray(sha),
                        **{f"choice_{k}": np.asarray(v) for k, v in choices.items()},
                        **{f"score_{k}": np.asarray(v) for k, v in scores.items()}))
                base.immutable(receipt, dict(identity=ident, fit=fitted, forecast=f, calibration=cal_table))
            r = study.read(receipt)
            if r["identity"] != ident:
                raise ValueError("Invalid substitution receipt")
            study.checked(r["forecast"])
            study.checked(r["fit"]["checkpoint"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(refs["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def score(key, data, source, control, refs, report):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(refs["folds"]):
        raise ValueError("Require sealed replacements")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    counts, by_city, rows, cases = Counter(), {}, [], {k: [] for k in MODES}
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        city = by_city.setdefault(held, Counter())
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], a[k])
            if z["registration"].item() != key:
                raise ValueError("Wrong registration on holdout")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v), anchor = data[n], truth[n], a["additive"][lo:hi]
                if d["city"] != held:
                    raise ValueError("Invalid held city")
                if len(eq.radii(d, seed)) != 2:
                    continue
                s = prepare(d, anchor, cut)
                if (digest(s) if s is not None else "") != z["digest"][i]:
                    raise ValueError("Menu changed after freeze")
                anchor_ok = eq.ceiling.eligible_additive(prev.metrics(anchor, d, y, v, cut))
                stat = Counter(two_slots=1, anchor_eligible=int(anchor_ok))
                row = dict(clip=str(n), city=held, seed=seed)
                if s is not None:
                    t = targets(d, s, anchor, y, v, cut)
                    stat.update(options=len(t)-1, local_available=int(t[:, 0].any()))
                for k in MODES:
                    j, threshold = int(z[f"choice_{k}"][i]), f["calibration"]["rules"][k]["threshold"]
                    match = bool(s is not None and t[j, 0])
                    active = bool(j > 0 and threshold is not None and z[f"score_{k}"][i] >= threshold)
                    stat.update({f"{k}_top": int(match), f"{k}_release": int(active), f"{k}_pass": int(active and match)})
                    row[k] = [0., 0.]
                    if active:
                        candidate = s[0][j]
                        g = cost.audit.direct_trajectory(candidate, anchor, d["cv"], y, v)
                        pair = av.frontier.pair_reduction(candidate[None], anchor, y, v)[0, 1]
                        met = prev.metrics(candidate, d, y, v, cut)
                        row[k] = [g[2], float(pair)] if g is not None else None
                        cases[k].append(dict(fold=name, match=match, anchor_eligible=anchor_ok,
                                             risk=met["risk"], gain=met["gain"], relative=row[k]))
                counts.update(stat)
                city.update(stat)
                rows.append(dict(row, **stat))
        beat("score", fold=name, scored=len(rows))
    if (counts["two_slots"], counts["anchor_eligible"]) != (
            report["counts"]["two_slots"], report["counts"]["anchor_eligible"]):
        raise ValueError("Original comparison roster changed")
    cc = {k: sum(f["calibration"]["rules"][k]["accepted"] for f in seal["folds"].values()) for k in MODES}
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run",
        counts={k: counts[k] for k in ("two_slots", "anchor_eligible", "options", "local_available",
            *(f"{k}_{z}" for k in MODES for z in ("top", "release", "pass")))},
        by_city={k: dict(v) for k, v in by_city.items()}, calibration_accepted=cc, releases=cases,
        calibration_counts={k: sum(f["calibration"]["counts"][k] for f in seal["folds"].values())
                            for k in ("scenes", "anchor_eligible", "pair_available",
                                      *(f"{k}_matched_top" for k in MODES))},
        paired_top1_change=av.interval(rows, lambda r: r["history_top"]-r["geometry_top"]),
        released_gain_vs_fixed={k: av.interval(rows, lambda r: r[k]) for k in MODES},
        gain_order=["trajectory_lower_pp", "pair_lower_native"], fitted_heads=2*len(seal["folds"]),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("counts", "calibration_accepted", "calibration_counts", "releases")}, indent=2))


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
        key, data, source, roles, inner, outer, control, refs, report = register()
        if phase == "fit":
            freeze(key, data, source, roles, inner, outer, control, refs)
        else:
            score(key, data, source, control, refs, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
