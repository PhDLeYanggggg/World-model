"""Isolate donor selection conditioning for the unchanged AV2 release forecast."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_projection_risk as old
from scripts.audit_m3w_av2_projection_risk_matched import counts

np, torch, study, base, av = old.np, old.torch, old.study, old.base, old.av
timing, rel, bank, model = old.timing, old.rel, old.bank, old.model
OUT, LOCAL = old.OUT.parent/"av2_selected_projection_risk_v1", old.LOCAL.parent/"av2_selected_projection_risk_v1"
FITS = ("uniform_one", "causal_selected")
ARMS = ("transferred_cost", "all_legal", *FITS)
STEPS, SAVE, BATCH = old.RISK_STEPS, old.SAVE, old.BATCH


def beat(phase, **kw):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


def register():
    k, data, src, roles, inner, outer, control, refs, parent, position, _ = old.register()
    path = old.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if (f["registration"] != k or any(f[x] for x in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened"))
            or s["independent_confirmation"] or s["reserved_outcomes_opened"]):
        raise ValueError("Require frozen source-excluded projections")
    p = OUT/"registration.json"
    base.immutable(p, dict(code=study.artifact(Path(__file__)), parent=study.artifact(path), arms=ARMS,
        fit="identical score head, optimizer, initialization and matched city-balanced query batches",
        causal_selected="the nested, source-excluded action model's original ordered recipients, before direction projection",
        uniform_one="one deterministic uniform ordered assignment per donor query keyed by seed and clip",
        all_legal="exact replay of prior gain-risk model using all ordered assignments",
        scope="same cached three-donor direction models and producer inputs; same four-donor frozen calibration and held paths",
        steps=STEPS, save_every=SAVE, batch=BATCH, seeds=rel.eq.SEEDS, original_2pct=True,
        primary="paired held fixed top five per fold, separate unchanged calibration release",
        independent_confirmation=False, reserved_outcomes_opened=False, threads=4, workers=0))
    return base.sha(p), data, src, roles, inner, outer, control, refs, parent, position, f, s


def uniform_choice(clip, seed, n):
    digest = hashlib.sha256(f"{seed}:{clip}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") % n


def load_direction(receipt, training, key, seed, width):
    ck = torch.load(study.checked(receipt["checkpoint"]), weights_only=True, map_location="cpu")
    if (ck["identity"] != receipt["identity"] or ck["identity"]["registration"] != key
            or ck["identity"]["training"] != training or ck["identity"]["seed"] != seed or ck["step"] != old.DIR_STEPS):
        raise ValueError("Wrong cached direction model")
    net = old.proj.head(width)
    net.load_state_dict(ck["net"])
    return net.eval()


def examples(data, truth, role, inner, refs, frozen):
    held, cal, donors, seed = (frozen["identity"][k] for k in ("held", "calibration", "donors", "seed"))
    nested = timing.old.donor_forecasts(inner, data, held, role, seed)
    if set(nested) != set(truth):
        raise ValueError("Wrong donor roster")
    groups = {k: {} for k in FITS}
    totals = {k: defaultdict(int) for k in (*FITS, "all_legal")}
    for city in donors:
        l = frozen["lineage"][city]
        tr = [c for c in donors if c != city]
        if set(l["training"]) != set(tr) or set(tr) & {city, held, cal} or l["target_city"] != city:
            raise ValueError("Wrong source-excluded lineage")
        a_ref = refs["references"][city]
        eq_ref = inner["models"][f"{role['inner'][city]}_{seed}"]
        if l["action"] != a_ref["checkpoint"] or l["eqmotion"] != eq_ref["forecast"]:
            raise ValueError("Donor producer changed")
        action, cut = timing.rel.pair.inner_action(a_ref, refs["identity"], city)
        net, samples = None, {k: [] for k in FITS}
        for n, (y, v) in truth.items():
            d = data[n]
            if d["city"] != city:
                continue
            a = timing.incumbent(d, seed, cut, nested[n], action, action_cut=cut)
            if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
                continue
            p, _, opts, chosen, _ = bank.candidate(d, a)
            actors = np.arange(len(p))
            if net is None:
                width = old.proj.inputs(d, p[:, 0], actors)[0].shape[1]
                r = l["direction"]
                net = load_direction(r, l["training"], frozen["identity"]["registration"], seed, width)
            projected = np.stack([rel.base_positions(net, d, p[:, j], actors) for j in range(2)], axis=1)
            np.testing.assert_allclose(np.linalg.norm(projected-d["cv"][:, None], axis=-1),
                                       np.linalg.norm(p-d["cv"][:, None], axis=-1), rtol=1e-8, atol=1e-8)
            x, scale = model.inputs(d, projected, opts, cut[0])
            gain, status = model.targets(d, projected, opts, y, v, cut[0], scale)
            exact, _ = bank.model.eligibility(d["cv"], d["ols4"], projected, y, v, cut[0], opts)
            if not np.array_equal(exact, status.all(1)):
                raise ValueError("Wrong qualification target")
            chosen_ix = dict(uniform_one=uniform_choice(n, seed, len(opts)), causal_selected=chosen)
            for k, ix in chosen_ix.items():
                samples[k].append((x[ix:ix+1], gain[ix:ix+1], status[ix:ix+1]))
            for k, ix in (*chosen_ix.items(), ("all_legal", slice(None))):
                lab = status[ix:ix+1] if isinstance(ix, (int, np.integer)) else status
                totals[k]["rows"] += len(lab)
                totals[k]["support"] += int(lab[:, 0].sum())
                totals[k]["support_and_risk"] += int((lab[:, 0] & lab[:, 1]).sum())
                totals[k]["qualified"] += int(lab.all(1).sum())
        for k in FITS:
            if not samples[k]:
                raise ValueError("No paired donor examples")
            groups[k][city] = old.batches(samples[k])
        if len(groups[FITS[0]][city][0]) != len(groups[FITS[1]][city][0]):
            raise ValueError("Matched query counts changed")
    if totals["all_legal"]["rows"] != frozen["fit"]["rows"]:
        raise ValueError("All-legal source roster did not replay")
    return groups, totals


def fit(ident, name, data):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+819)
    width = next(iter(data.values()))[0].shape[-1]
    net = model.head(width)
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume receipt")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, trace = ck["step"], ck["trace"]
    for step in range(start+1, STEPS+1):
        sampled = [(r, rng.integers(len(r[0]), size=BATCH//len(data))) for r in data.values()]
        x, gain, labels = (torch.cat([r[i][ix] for r, ix in sampled]) for i in range(3))
        loss, terms = model.loss(net(x), gain, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(t.detach()) for t in terms])
            q = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), q)
            q.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), trace=trace, rows=sum(len(r[0]) for r in data.values()))


def freeze(key, data, src, roles, inner, outer, control, refs, parent, old_freeze):
    done = {}
    for name, ref in old_freeze["folds"].items():
        r = study.read(study.checked(ref))
        ident = dict(r["identity"], registration=key, source=ref)
        held, cal, donors, seed = (ident[k] for k in ("held", "calibration", "donors", "seed"))
        role = roles[held]
        if ident["parent"] != parent["folds"][name] or donors != role["donors"] or cal != role["calibration"] or held in [cal, *donors] or cal in donors:
            raise ValueError("Wrong source roles")
        p = LOCAL/f"{name}.json"
        if not p.exists():
            truth = old.prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            groups, train = examples(data, truth, role, inner, refs["folds"][name], r)
            fitted = {k: fit(dict(ident, arm=k), f"{name}_{k}", groups[k]) for k in FITS}
            f = study.read(study.checked(ident["parent"]))
            pp = study.read(study.checked(f["identity"]["parent"]))
            rows = {}
            for side in ("calibration", "held"):
                rows[side] = []
                for (n, d, _, ix, digest, _), z, q in zip(rel.inputs(data, pp, control["folds"][name], outer["folds"][name], side),
                                                         f["rows"][side], r["rows"][side], strict=True):
                    if not n == z["clip"] == q["clip"] or digest != z["hash"] or ix.tolist() != z["edited"]:
                        raise ValueError("Wrong frozen forecast")
                    path = d["cv"].copy()
                    path[ix] = z["paths"][0]
                    s = dict(transferred_cost=q["score"]["transferred_cost"], all_legal=q["score"]["gain_risk"])
                    s.update({k: old.predict(net, d, path, ix, f["identity"]["cut"])["gain_risk"] for k, (net, _) in fitted.items()})
                    rows[side].append(dict(clip=str(n), city=d["city"], score=s))
            base.immutable(p, dict(identity=ident, rows=rows, training=train, fits={k: t for k, (_, t) in fitted.items()}))
        z = study.read(p)
        if z["identity"] != ident or any(z["fits"][k]["rows"] != z["training"][k]["rows"] for k in FITS):
            raise ValueError("Wrong selected-fit receipt")
        for k in FITS:
            study.checked(z["fits"][k]["checkpoint"])
        done[name] = study.artifact(p)
        beat("freeze", completed=len(done), total=len(old_freeze["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, calibration_labels_used=False,
                                          held_labels_used=False, reserved_outcomes_opened=False))


def score(key, parent, baseline):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or any(f[k] for k in ("calibration_labels_used", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require outcome-free score freeze")
    rules, training = {}, defaultdict(lambda: defaultdict(int))
    frozen = {k: study.read(study.checked(r)) for k, r in f["folds"].items()}
    for name, z in frozen.items():
        for arm, nums in z["training"].items():
            for m, n in nums.items():
                training[arm][m] += n
        rows = list(old.prior.role_rows(z, parent["folds"][name], name, "calibration"))
        rules[name] = {k: timing.old.action.prior.calibration([dict(score=r["scores"][k], strict=r["result"]["eligible"],
            add_ok=True) for r in rows]) for k in ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, held_receipts_used=False))
    all_held, rank, cells = [], defaultdict(list), {k: [] for k in (1, 5, 10)}
    for name, z in frozen.items():
        rows = list(old.prior.role_rows(z, parent["folds"][name], name, "held"))
        for a in ARMS:
            rank[a].append(old.ranks(rows, a))
        for k, v in cells.items():
            if rows:
                v.append(dict(city=z["identity"]["held"], clip=f"budget_{k}", n=min(k, len(rows)),
                              arms={a: counts(rows, a, k) for a in ARMS}))
        for r in rows:
            r["release"] = {a: rules[name][a]["threshold"] is not None and r["scores"][a] >= rules[name][a]["threshold"] for a in ARMS}
        all_held.extend(rows)
    held = {a: old.prior.held_metrics(all_held, a) for a in ARMS}
    for a, b in (("transferred_cost", "transferred_cost"), ("all_legal", "gain_risk")):
        if held[a] != baseline["held"][b]:
            raise ValueError("Matched comparator did not replay")
    forced = sum(r["result"]["eligible"] for r in all_held)
    if len(all_held) != baseline["views"] or forced != baseline["forced_qualifies"]:
        raise ValueError("Frozen position qualification changed")
    matched = {}
    for k, v in cells.items():
        fields = next(iter(v[0]["arms"].values()))
        matched[k] = dict(n=sum(r["n"] for r in v), counts={a: {m: sum(r["arms"][a][m] for r in v) for m in fields} for a in ARMS},
            selected_minus={a: {m: av.interval(v, lambda r: 100.*(r["arms"]["causal_selected"][m]-r["arms"][a][m])/r["n"])
                         for m in fields} for a in ARMS[:-1]})
    result = dict(views=len(all_held), scenarios=len({r["clip"] for r in all_held}), forced_qualifies=forced, training=training,
        fold_mean_ap={a: float(np.mean([z["ap"] for z in zs if z["ap"] is not None])) for a, zs in rank.items()},
        matched=matched, held=held, calibration_pass={a: sum(r[a]["accepted"] for r in rules.values()) for a in ARMS},
        freeze=study.artifact(p), calibration=study.artifact(cp), result_source="fresh_run", outcome_source="cached_verified",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, inner, outer, control, refs, parent, _, old_freeze, baseline = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent, old_freeze)
        else:
            score(key, parent, baseline)


if __name__ == "__main__":
    main()
