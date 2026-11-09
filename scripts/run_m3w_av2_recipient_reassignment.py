"""Separate recipient choice from direction at an exact two-profile budget."""

import argparse
from collections import Counter
import fcntl
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_av2_steered_reference_ceiling as ceiling
from src.world_model import m3w_av2_recipient_reassignment as model


stage = ceiling.stage
np, torch, av, eq, prev, study, base = stage.np, stage.torch, stage.av, stage.eq, stage.prev, stage.study, stage.base
OUT, LOCAL = stage.OUT.parent/"av2_recipient_reassignment_v1", stage.LOCAL.parent/"av2_recipient_reassignment_v1"
ARMS = ("anchor", *model.ARMS)
CAPACITY = ("fixed_free", "swap_free", "reassign_free", "reassign_transport", "reassign_45")


def beat(phase, n):
    print(dict(phase=phase, completed=n, total=18), flush=True)


def register():
    _, data, source, _, _, control, _, _ = stage.register()
    p = stage.OUT/"summary.json"
    r = study.read(p)
    cap = stage.OUT/"direction_ceiling.json"
    parent = study.read(cap)
    f = study.read(study.checked(r["freeze"]))
    if r["reserved_outcomes_opened"] or parent["reserved_outcomes_opened"] or f["held_labels_used"] or parent["score"] != study.artifact(p):
        raise ValueError("Require complete and frozen development forecasts")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(p), direction_ceiling=study.artifact(cap), arms=ARMS, capacities=CAPACITY,
        scope="same two profiles with unchanged radial magnitude at each step, and same candidate rows and label masks",
        transport="rotate a profile from its original actor's observed heading to the new actor's observed heading; no scale change",
        selection="past prefix replay: fixed recipients, unrestricted additive, predicted easy support, and same support plus pair term",
        pair_weight=model.PAIR_WEIGHT, support_cut="positive CV prefix error within mean-step-scaled original cut on at least two prefixes",
        primary="exact recipient ceiling vs fixed and held standalone eligibility for support-constrained additive vs pair",
        no_new_training=True, original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(reg), data, source, control, f, r, parent


def candidate(d, a):
    p, slots = model.bank(d["cv"], d["h"], a)
    opts = model.assignments(len(p))
    idx = np.flatnonzero((opts == slots).all(1)).item()
    return p, slots, opts, idx, hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()


def freeze(key, data, control, parent):
    done = {}
    for name, f in parent["folds"].items():
        mark = LOCAL/f"{name}.json"
        ident = dict(registration=key, fold=f["identity"])
        if not mark.exists():
            held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
            with np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a:
                rows = []
                for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
                    d = data[n]
                    if d["city"] != held:
                        raise ValueError("Wrong held city")
                    if len(eq.radii(d, seed)) != 2:
                        continue
                    p, slots, opts, idx, digest = candidate(d, a["additive"][lo:hi])
                    choice, support = model.select(d["h"], d["cv"], p, slots, opts, cut[0])
                    rows.append(dict(clip=str(n), hash=digest, choices=[idx, *choice.tolist()], predicted_easy_available=support))
            base.immutable(mark, dict(identity=ident, rows=rows))
        r = study.read(mark)
        if r["identity"] != ident:
            raise ValueError("Wrong fold receipt")
        done[name] = study.artifact(mark)
        beat("freeze", len(done))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False, reserved_outcomes_opened=False))


def exact_eligibility(d, p, opts, y, v, cut):
    eligible, supported = model.eligibility(d["cv"], d["ols4"], p, y, v, cut[0], opts)
    for i in np.flatnonzero(eligible):
        m = prev.metrics(model.render(d["cv"], p, opts[i]), d, y, v, cut)
        if not eq.ceiling.eligible_additive(m):
            raise ValueError("Standalone witness failed scalar replay")
    return eligible, supported


def fold_score(data, truth, f, cf, fixed):
    held, seed, cut = (f["identity"]["fold"][k] for k in ("held", "seed", "cut"))
    rows = []
    with np.load(study.checked(cf["forecast"]), allow_pickle=False) as a:
        for n, lo, hi in zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True):
            d = data[n]
            if d["city"] != held:
                raise ValueError("Wrong held city")
            if len(eq.radii(d, seed)) != 2:
                continue
            q = f["rows"][len(rows)]
            anchor = a["additive"][lo:hi]
            p, slots, opts, idx, digest = candidate(d, anchor)
            if q["clip"] != str(n) or q["hash"] != digest or q["choices"][0] != idx:
                raise ValueError("Recipient bank changed after freeze")
            y, v = truth[n]
            cv, ols = d["cv"], d["ols4"]
            equal = (np.sort(opts, axis=1) == slots).all(1)
            exact, support = exact_eligibility(d, p, opts, y, v, cut)
            free, s = exact_eligibility(d, ceiling.best_direction(cv[:, None], ols[:, None], p, y[:, None], v[:, None], np.pi), opts, y, v, cut)
            bounded, b = exact_eligibility(d, ceiling.best_direction(cv[:, None], ols[:, None], p, y[:, None], v[:, None], np.pi/4.), opts, y, v, cut)
            if not np.array_equal(s, support) or not np.array_equal(b, support) or (exact & ~bounded).any() or (bounded & ~free).any():
                raise ValueError("Relaxation lost support or a feasible assignment")
            row = dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), predicted_easy_available=q["predicted_easy_available"],
                       capacity=dict(fixed_free=bool(free[idx]), swap_free=bool(free[equal].any()), reassign_free=bool(free.any()),
                                     reassign_transport=bool(exact.any()), reassign_45=bool(bounded.any())),
                       fixed_supported=bool(support[idx]), reassign_supported=bool(support.any()), arms={})
            options = [model.render(cv, p, opts[c]) for c in q["choices"]]
            np.testing.assert_array_equal(options[0], anchor)
            for k, c, path in zip(ARMS, q["choices"], options, strict=True):
                m = prev.metrics(path, d, y, v, cut)
                eligible = eq.ceiling.eligible_additive(m)
                if eligible != bool(exact[c]):
                    raise ValueError("Causal eligibility failed exact scalar replay")
                risk = m["risk"]
                lower = stage.prior.cost.audit.direct_trajectory(path, anchor, cv, y, v)
                pair = av.frontier.pair_reduction(path[None], options[-2], y, v)[0, 1]
                row["arms"][k] = dict(eligible=eligible, risk=risk["supported"], easy_defined=risk["completion_easy_risk_upper"] is not None,
                    signed_easy_over=(risk["known_easy_net_degradation"] or 0.) > .02, gain=m["gain"], ade=m["ade"], fde=m["fde"], tail=m["tail"],
                    lower_vs_anchor=lower[2] if lower is not None else None, pair_vs_support=float(pair))
            if bool(fixed[str(n)]["arms"]["anchor"]["eligible"]) != row["arms"]["anchor"]["eligible"]:
                raise ValueError("Frozen original eligibility changed")
            rows.append(row)
    if len(rows) != len(f["rows"]):
        raise ValueError("Scoring roster changed")
    return rows


def summarize(rows, parent):
    count = lambda z: {k: sum(r["capacity"][k] for r in z) for k in CAPACITY}
    caps = count(rows)
    if caps["fixed_free"] != parent["ceilings"]["free"] or sum(r["fixed_supported"] for r in rows) != parent["fixed_easy_supported"]:
        raise ValueError("Previous exact fixed-direction ceiling did not replay")
    arms = {k: dict(**{m: sum(r["arms"][k][m] for r in rows) for m in ("eligible", "risk", "easy_defined", "signed_easy_over")},
        complete_eligible=sum(r["arms"][k]["eligible"] for r in rows if r["complete"]),
        worst_p95=max(r["arms"][k]["tail"] for r in rows if r["arms"][k]["tail"] is not None),
        **{m: av.interval(rows, lambda r: r["arms"][k][m]) for m in ("gain", "ade", "fde", "lower_vs_anchor", "pair_vs_support")}) for k in ARMS}
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
        capacity=caps, complete_capacity=count([r for r in rows if r["complete"]]),
        supported={k: sum(r[k] for r in rows) for k in ("fixed_supported", "reassign_supported", "predicted_easy_available")},
        arms=arms, paired_eligibility={a+"_minus_"+b: av.interval(rows, lambda r: int(r["arms"][a]["eligible"])-int(r["arms"][b]["eligible"]))
            for a, b in (("history_gain", "fixed_actors"), ("history_support", "history_gain"), ("history_joint", "history_support"))},
        capacity_by_city={g: count([r for r in rows if r["city"] == g]) for g in sorted({r["city"] for r in rows})})


def score(key, data, source, control, parent, bound, replay):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["held_labels_used"] or f["reserved_outcomes_opened"]:
        raise ValueError("Require sealed recipient choices")
    rows, truth = [], None
    for j, (name, ref) in enumerate(f["folds"].items(), 1):
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay may not reopen labels")
            if truth is None:
                truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
            prior = study.read(stage.LOCAL/f"{name}.score.json")
            if prior["freeze"] != parent["folds"][name]:
                raise ValueError("Wrong prior fixed reference")
            z = fold_score(data, truth, study.read(study.checked(ref)), control["folds"][name],
                           {x["clip"]: x for x in prior["rows"]})
            base.immutable(mark, dict(freeze=ref, rows=z))
        z = study.read(mark)
        if z["freeze"] != ref:
            raise ValueError("Incorrect score receipt")
        rows.extend(z["rows"])
        beat("replay" if replay else "score", j)
    result = summarize(rows, bound)
    base.immutable(OUT/"summary.json", dict(**result, freeze=study.artifact(p), result_source="fresh_run",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False,
        release_tested=False, no_new_training=True))
    print(study.json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, source, control, parent, _, bound = register()
        if phase == "freeze":
            freeze(key, data, control, parent)
        else:
            score(key, data, source, control, parent, bound, replay=phase == "replay")


if __name__ == "__main__":
    main()
