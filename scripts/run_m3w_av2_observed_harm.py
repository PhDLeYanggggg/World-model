"""Add a supported observed-harm score to the frozen AV2 position policy."""

import argparse
from collections import Counter, defaultdict
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_projection_support as support
from scripts import run_m3w_av2_projection_risk as old
from scripts.run_m3w_av2_selected_projection_risk import load_direction
from scripts.audit_m3w_av2_projection_risk_matched import weights
from src.world_model import m3w_av2_observed_harm as model

np, torch, study, base, av = support.np, support.torch, support.study, support.base, support.av
rel, timing, bank = old.rel, old.timing, old.bank
OUT, LOCAL = support.OUT.parent/"av2_observed_harm_v1", support.LOCAL.parent/"av2_observed_harm_v1"
ARMS = ("transferred_cost", "actor_support", "conditional_harm")
STEPS, SAVE, BATCH = 400, 50, 96


def beat(phase, **kw):
    r = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **kw)
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(r)+"\n")
    p.replace(LOCAL/"heartbeat.json")
    print(r, flush=True)


def register():
    k, data, src, roles, outer, control, parent, _, _, _ = support.register()
    path = support.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != k or f["held_labels_used"] or s["reserved_outcomes_opened"] or s["independent_confirmation"]:
        raise ValueError("Require the scored development support study")
    k2, _, _, _, inner, _, _, refs, _, _, _ = old.register()
    nested_path = old.OUT/"summary.json"
    n = study.read(study.checked(study.read(nested_path)["freeze"]))
    if n["registration"] != k2 or n["held_labels_used"] or n["reserved_outcomes_opened"]:
        raise ValueError("Require sealed source-excluded direction producers")
    p = OUT/"registration.json"
    base.immutable(p, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(path), nested=study.artifact(nested_path), arms=ARMS, seeds=rel.eq.SEEDS,
        steps=STEPS, save_every=SAVE, batch=BATCH, threads=4, workers=0,
        target="known all and easy positive harm on source-excluded projected options with positive edited-easy support; "
               "observed exceedance classifier and auxiliary log ratios; absent support does not count as zero harm",
        score="same frozen support and gain/harm score, multiplied by conditional no-observed-harm probability",
        primary="paired fixed-five qualification and observed harm; actual original calibration release separate",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), data, src, roles, inner, outer, control, refs, parent, f, n, s


def examples(data, truth, role, inner, ref, frozen):
    held, cal, donors, seed = (frozen["identity"][k] for k in ("held", "calibration", "donors", "seed"))
    raw = timing.old.donor_forecasts(inner, data, held, role, seed)
    if set(raw) != set(truth):
        raise ValueError("Wrong donor roster")
    groups = {}
    for city in donors:
        l = frozen["lineage"][city]
        if (set(l["training"]) != set(donors)-{city} or set(l["training"]) & {held, cal, city}
                or l["target_city"] != city or l["action"] != ref["references"][city]["checkpoint"]
                or l["eqmotion"] != inner["models"][f"{role['inner'][city]}_{seed}"]["forecast"]):
            raise ValueError("Wrong direction or producer lineage")
        action, cut = timing.rel.pair.inner_action(ref["references"][city], ref["identity"], city)
        rows, net = [], None
        for n, (y, v) in truth.items():
            d = data[n]
            if d["city"] != city:
                continue
            a = timing.incumbent(d, seed, cut, raw[n], action, action_cut=cut)
            if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
                continue
            p, _, opts, _, _ = bank.candidate(d, a)
            ids = np.arange(len(p))
            if net is None:
                width = old.proj.inputs(d, p[:, 0], ids)[0].shape[1]
                net = load_direction(l["direction"], l["training"], frozen["identity"]["registration"], seed, width)
            paths = np.stack([rel.base_positions(net, d, p[:, j], ids) for j in range(2)], axis=1)
            mask, bad, severity = model.targets(d["cv"], paths, opts, y, v, cut[0])
            _, exact = bank.model.eligibility(d["cv"], d["ols4"], paths, y, v, cut[0], opts)
            if not np.array_equal(mask, exact):
                raise ValueError("Support supervision changed")
            if mask.any():
                x, _ = old.model.inputs(d, paths, opts, cut[0])
                rows.append((x[mask], bad[mask], severity[mask]))
        if not rows:
            raise ValueError("No supported donor options")
        groups[city] = old.batches(rows)
    return dict(sorted(groups.items()))


def fit(ident, name, data):
    torch.manual_seed(ident["seed"])
    rng = np.random.default_rng(ident["seed"]+1189)
    net = model.head(next(iter(data.values()))[0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    p = LOCAL/f"{name}.pt"
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, weights_only=True, map_location="cpu")
        if ck["identity"] != ident or ck["step"] not in range(SAVE, STEPS+1, SAVE):
            raise ValueError("Wrong resume identity")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng"]
        start, trace = ck["step"], ck["trace"]
    for step in range(start+1, STEPS+1):
        b = [(r, rng.integers(len(r[0]), size=BATCH//len(data))) for r in data.values()]
        x, y, z = (torch.cat([r[k][ix] for r, ix in b]) for k in range(3))
        loss, c, m = model.loss(net(x), y, z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % SAVE == 0:
            trace.append([float(c.detach()), float(m.detach())])
            tmp = p.with_suffix(".pending")
            torch.save(dict(identity=ident, step=step, net=net.state_dict(), optimizer=opt.state_dict(),
                            rng=rng.bit_generator.state, trace=trace), tmp)
            tmp.replace(p)
            beat("fit", fold=name, step=step, total=STEPS)
    return net.eval(), dict(checkpoint=study.artifact(p), supported=sum(len(r[1]) for r in data.values()),
        observed_harm=sum(int(r[1].sum()) for r in data.values()), trace=trace)


@torch.inference_mode()
def predict(net, d, path, ix, cut):
    x, _ = old.model.inputs(d, np.repeat(path[:, None], 2, axis=1), ix[None], cut)
    return float(net(torch.from_numpy(x))[0, 0].sigmoid())


def freeze(key, data, src, roles, inner, outer, control, refs, parent, sf, nf):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        ss = study.read(study.checked(sf["folds"][name]))
        nn = study.read(study.checked(nf["folds"][name]))
        held, cal, donors = (f["identity"][k] for k in ("held", "calibration", "donors"))
        if held in [cal, *donors] or cal in donors or ss["identity"]["source"] != ref or nn["identity"]["parent"] != ref:
            raise ValueError("Wrong support or direction role")
        ident = dict(f["identity"], registration=key, source=ref)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            dy = old.prev.load_targets(src, data, [n for n, d in data.items() if d["city"] in donors])
            net, r = fit(ident, name, examples(data, dy, roles[held], inner, refs["folds"][name], nn))
            rows = {}
            for role in ("calibration", "held"):
                rows[role] = []
                for (n, d, ix, p), q in zip(support.frozen_paths(data, f, control["folds"][name], outer["folds"][name], role),
                                             ss["rows"][role], strict=True):
                    if n != q["clip"]:
                        raise ValueError("Frozen score roster changed")
                    s = {a: q["score"][a] for a in ARMS[:2]}
                    risk = predict(net, d, p, ix, ident["cut"][0])
                    s["conditional_harm"] = s["actor_support"]*(1.-risk)
                    rows[role].append(dict(clip=str(n), city=d["city"], score=s, harm_probability=risk))
            base.immutable(mark, dict(identity=ident, rows=rows, fit=r))
        z = study.read(mark)
        if z["identity"] != ident:
            raise ValueError("Wrong frozen result")
        study.checked(z["fit"]["checkpoint"])
        done[name] = study.artifact(mark)
        beat("freeze", completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False,
        calibration_labels_used=False, reserved_outcomes_opened=False))


def cells(rows, k, oracle=False):
    result = {}
    fields = ("qualified", "risk_fail", "unsupported", "observed_harm")
    for a in (*ARMS, *(("known_no_harm",) if oracle else ())):
        score = np.asarray([r["scores"][a] if a in ARMS else
            (r["scores"]["actor_support"] if r["support"] and not r["harm"] else -np.inf) for r in rows])
        w = weights(score, k)
        y = np.array([(r["result"]["eligible"], not r["result"]["risk"], not r["support"], r["harm"]) for r in rows])
        result[a] = dict(zip(fields, map(float, w@y), strict=True))
    return result


def summarize_cells(rows):
    out = {a: {k: sum(r["arms"][a][k] for r in rows) for k in vals} for a, vals in rows[0]["arms"].items()}
    return dict(n=sum(r["n"] for r in rows), counts=out, minus_actor_support_pp={a: {k: av.interval(rows,
        lambda r: 100.*(r["arms"][a][k]-r["arms"]["actor_support"][k])/r["n"]) for k in vals}
        for a, vals in out.items() if a != "actor_support"})


def score(key, parent, previous):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or any(f[k] for k in ("held_labels_used", "calibration_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require forecasts frozen before outcome access")
    frozen = {n: study.read(study.checked(r)) for n, r in f["folds"].items()}
    rules = {}
    for n, z in frozen.items():
        r = list(support.prior.role_rows(z, parent["folds"][n], n, "calibration"))
        rules[n] = {a: timing.old.action.prior.calibration([dict(score=q["scores"][a], strict=q["result"]["eligible"],
                    add_ok=True) for q in r]) for a in ARMS}
    cp = OUT/"calibration.json"
    base.immutable(cp, dict(freeze=study.artifact(p), rules=rules, held_receipts_used=False))
    held, train, top, diagnostic, aps = [], Counter(), [], [], defaultdict(list)
    for n, z in frozen.items():
        train.update({k: z["fit"][k] for k in ("supported", "observed_harm")})
        pf = parent["folds"][n]
        annotated = study.read(support.LOCAL/f"{n}.support.json")
        if annotated["parent"] != pf:
            raise ValueError("Wrong support receipt")
        rows = list(support.prior.role_rows(z, pf, n, "held"))
        for r, q, a in zip(rows, z["rows"]["held"], annotated["cells"], strict=True):
            if r["clip"] != a["clip"]:
                raise ValueError("Wrong annotation row")
            r.update(support=a["support"], cause=a["cause"], harm=a["cause"] == "observed_positive_harm",
                     probability=q["harm_probability"])
            r["release"] = {k: rules[n][k]["threshold"] is not None and r["scores"][k] >= rules[n][k]["threshold"] for k in ARMS}
        for a in ("actor_support", "conditional_harm"):
            x = [dict(scores={"p": -r["scores"][a]}, result={"eligible": r["harm"]}) for r in rows if r["support"]]
            aps[a].append(support.ranks(x, "p")["ap"])
        x = [dict(scores={"p": r["probability"]}, result={"eligible": r["harm"]}) for r in rows if r["support"]]
        aps["harm_probability"].append(support.ranks(x, "p")["ap"])
        if rows:
            k = min(5, len(rows))
            top.append(dict(city=z["identity"]["held"], clip="top5", n=k, arms=cells(rows, k)))
            k = min(5, sum(r["support"] and not r["harm"] for r in rows))
            if k:
                diagnostic.append(dict(city=z["identity"]["held"], clip="diagnostic", n=k, arms=cells(rows, k, oracle=True)))
        held.extend(rows)
    h = {a: support.prior.held_metrics(held, a) for a in ARMS}
    if (len(held) != previous["views"] or sum(r["support"] for r in held) != previous["support_views"]
            or any(h[a] != previous["held"][a] for a in ARMS[:2])):
        raise ValueError("Frozen support comparator did not replay")
    supported = [r for r in held if r["support"]]
    result = dict(views=len(held), scenarios=len({r["clip"] for r in held}), train=train, held=h,
        support_views=len(supported), observed_harm_views=sum(r["harm"] for r in supported),
        conditional_harm_ap={a: float(np.mean([b for b in x if b is not None])) for a, x in aps.items()},
        conditional_harm_brier=av.interval(supported, lambda r: (r["probability"]-r["harm"])**2),
        matched_top5=summarize_cells(top), known_no_harm_diagnostic=summarize_cells(diagnostic),
        calibration_pass={a: sum(v[a]["accepted"] for v in rules.values()) for a in ARMS},
        release_causes={a: Counter(r["cause"] for r in held if r["release"][a]) for a in ARMS},
        freeze=study.artifact(p), calibration=study.artifact(cp), result_source="fresh_run",
        outcome_source="cached_verified", original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False)
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
        key, data, src, roles, inner, outer, control, refs, parent, sf, nf, previous = register()
        if phase == "fit":
            freeze(key, data, src, roles, inner, outer, control, refs, parent, sf, nf)
        else:
            score(key, parent, previous)


if __name__ == "__main__":
    main()
