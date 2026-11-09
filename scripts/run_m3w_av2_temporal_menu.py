"""Measure whether changing edit timing improves fixed-budget AV2 candidate supply."""

import argparse
from collections import Counter
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import sys
from time import time

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_margin_assignment as assign
from src.world_model import m3w_av2_temporal_menu as model

np, torch, av, prev, eq, base, study, old = (
    assign.np, assign.torch, assign.av, assign.prev, assign.eq, assign.base, assign.study, assign.old)
OUT, LOCAL = assign.OUT.parent/"av2_temporal_menu_v1", assign.LOCAL.parent/"av2_temporal_menu_v1"
MODES = model.MODES
CHUNK = 96


def beat(phase, **fields):
    p = LOCAL/"heartbeat.pending"
    p.write_text(study.json.dumps(dict(phase=phase, pid=os.getpid(), epoch=time(), **fields))+"\n")
    p.replace(LOCAL/"heartbeat.json")


def register():
    parent, data, source, roles, _, outer, control, _ = assign.register()
    p = assign.OUT/"summary.json"
    report = study.read(p)
    seal = study.read(study.checked(report["freeze"]))
    if (seal["registration"] != parent or report["registration"] != parent or seal["held_labels_used"]
            or report["reserved_outcomes_opened"] or set(seal["folds"]) != set(outer["folds"])):
        raise ValueError("Require the corrected AV2 capacity and sealed city-excluded folds")
    reg = OUT/"registration.json"
    base.immutable(reg, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(p), parent_freeze=report["freeze"], modes=list(MODES),
        timing="each CV-relative edit offset multiplied by a linear .25 to 1 window or its reverse, then restored to its own raw peak",
        comparison="same recipients available, same original two peaks, same CV and OLS4, "
                   "same-size early and late replacement banks plus their union with the base bank",
        frozen_action="identical additive checkpoint and score on each bank; no fitting, no threshold tuned on held cities",
        calibration="standalone reference only; every nonempty score prefix must pass both motion floors and all original 2 percent risks",
        oracle="fixed-reference pair capacity and flexible reference/two-path capacity, evaluated separately from frozen action scores",
        units="native coordinates, 8 observation steps and 12 forecast steps, city bootstrap after averaging seeds by scenario",
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False, no_new_fits=True))
    return base.sha(reg), data, source, roles, outer, control, seal, report


def menu(d, seed, cut, raw):
    bank = eq.expanded(d, raw)
    wide = model.expanded(d["cv"], bank)
    m = old.action.menu(dict(d, bank=wide), seed, cut)
    if m is None:
        return None, None
    keep = model.membership(m["col"][m["opts"]], bank.shape[1])
    if not all(v.any() for v in keep.values()):
        raise ValueError("No option in a same-size bank")
    if len({int(keep[k].sum()) for k in MODES[:-1]}) != 1:
        raise ValueError("Temporal reshaping changed peak eligibility")
    return m, keep


def render(d, m, ix):
    p = av.slate.assemble(d["cv"], m["bank"], m["row"], m["col"], m["opts"][ix])
    actual = np.sort(av.motion.peak(p-d["cv"]), axis=1)
    np.testing.assert_array_equal((actual > 0.).sum(1), np.full(len(p), 2))
    np.testing.assert_allclose(actual[:, -2:], np.broadcast_to(np.sort(m["radii"]), (len(p), 2)), atol=1e-8, rtol=1e-8)
    return p


def digest(m):
    h = hashlib.sha256()
    for k in ("bank", "row", "col", "opts"):
        h.update(np.ascontiguousarray(m[k]).tobytes())
    return h.hexdigest()


@torch.inference_mode()
def choose(action, d, seed, cut, raw):
    m, keep = menu(d, seed, cut, raw)
    if m is None:
        return None, None, None
    z = action(torch.from_numpy(m["x"])).numpy()
    unit = m["unit"][m["row"]]
    picks, scores = {}, {}
    for k in MODES:
        at = np.flatnonzero(keep[k])
        _, order, scores[k] = old.action.rank(z, m["opts"][at], unit)
        picks[k] = int(at[order[0]])
    return picks, scores, m


def calibrate(action, forecaster, data, truth, seed, cut):
    raw, offsets = eq.predict(forecaster, data, list(truth))
    rows = {k: [] for k in MODES}
    for n, lo, hi in zip(truth, offsets[:-1], offsets[1:], strict=True):
        d, (y, v) = data[n], truth[n]
        picks, scores, m = choose(action, d, seed, cut, raw[lo:hi])
        if picks is None:
            continue
        p = render(d, m, list(picks.values()))
        for k, path in zip(MODES, p, strict=True):
            ok = eq.ceiling.eligible_additive(prev.metrics(path, d, y, v, cut))
            rows[k].append(dict(score=scores[k], strict=ok, add_ok=True))
    return {k: old.action.prior.calibration(rows[k]) for k in MODES}


def freeze(key, data, source, roles, outer, control, parent):
    done = {}
    for held, role in roles.items():
        cal, donors = role["calibration"], role["donors"]
        cy = prev.load_targets(source, data, [n for n, d in data.items() if d["city"] == cal and d["added"]])
        for seed in eq.SEEDS:
            name = f"{held}_{seed}"
            producer, fixed = outer["folds"][name], control["folds"][name]
            cut = parent["folds"][name]["identity"]["cut"]
            ident = dict(registration=key, held=held, calibration=cal, donors=donors, seed=seed, cut=cut)
            if any(ident[k] != fixed["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut")):
                raise ValueError("Frozen action roles differ")
            receipt = LOCAL/f"{name}.json"
            if not receipt.exists():
                action = assign.prior.load_action(fixed)
                table = calibrate(action, old.load_outer(producer), data, cy, seed, cut)
                with (np.load(study.checked(producer["forecast"]), allow_pickle=False) as raw,
                      np.load(study.checked(fixed["forecast"]), allow_pickle=False) as a):
                    for k in ("names", "offsets"):
                        np.testing.assert_array_equal(raw[k], a[k])
                    picks, scores, digests = {k: [] for k in MODES}, {k: [] for k in MODES}, []
                    for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                        d = data[n]
                        if d["city"] != held:
                            raise ValueError("Wrong held city")
                        ix, conf, m = choose(action, d, seed, cut, raw["raw_eq"][lo:hi])
                        if ix is not None:
                            np.testing.assert_array_equal(render(d, m, [ix["base"]])[0], a["additive"][lo:hi])
                        for k in MODES:
                            picks[k].append(ix[k] if ix is not None else -1)
                            scores[k].append(conf[k] if conf is not None else -1.)
                        digests.append(digest(m) if m is not None else "")
                    bank = av.turn.exp.previous.save_bank(LOCAL/f"{name}.npz", dict(
                        names=raw["names"], offsets=raw["offsets"], registration=key, digest=np.asarray(digests),
                        **{f"choice_{k}": np.asarray(v) for k, v in picks.items()},
                        **{f"score_{k}": np.asarray(v) for k, v in scores.items()}))
                base.immutable(receipt, dict(identity=ident, forecast=bank, calibration=table,
                                             action=fixed["action"]["checkpoint"], parent_bank=producer["forecast"]))
            r = study.read(receipt)
            if r["identity"] != ident or r["action"] != fixed["action"]["checkpoint"] or r["parent_bank"] != producer["forecast"]:
                raise ValueError("Invalid frozen temporal menu receipt")
            study.checked(r["forecast"])
            done[name] = r
            beat("freeze", fold=name, completed=len(done), total=len(parent["folds"]))
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, held_labels_used=False))


def exact_pair(candidates, anchor, y, valid, mass):
    if not len(candidates) or mass <= 0.:
        return None
    norm = lambda x: np.linalg.norm(x, axis=-1)
    ref = norm(anchor-y)
    for lo in range(0, len(candidates), CHUNK):
        p = candidates[lo:lo+CHUNK]
        traj = np.where(valid, ref-norm(p-y), -norm(p-anchor)).sum((1, 2))
        possible = np.flatnonzero(traj >= -1e-10*mass)
        if len(possible):
            pair = av.frontier.pair_reduction(p[possible], anchor, y, valid)[:, 1]
            good = np.flatnonzero(pair > 1e-10)
            if len(good):
                j = possible[good[0]]
                return lo+int(j)
    return None


def witness(d, candidates, anchor, y, v, cut, mass):
    j = exact_pair(candidates, anchor, y, v, mass)
    if j is None:
        return False
    if not all(eq.ceiling.eligible_additive(prev.metrics(p, d, y, v, cut)) for p in (candidates[j], anchor)):
        raise ValueError("Vectorized qualification disagrees with scalar witness")
    return True


def counts(d, m, keep, y, v, cut, fixed, selected):
    _, _, qualified = assign.labels(d, m, y, v, cut)
    valid = qualified.numpy()
    idx = np.flatnonzero(valid)
    safe = np.concatenate([render(d, m, idx[i:i+CHUNK]) for i in range(0, len(idx), CHUNK)]) if len(idx) else None
    fixed_ok = eq.ceiling.eligible_additive(prev.metrics(fixed, d, y, v, cut))
    mass = float(np.where(v, np.linalg.norm(d["cv"]-y, axis=-1), 0.).sum())
    out = {}
    for k in MODES:
        j = selected[k]
        if not keep[k][j]:
            raise ValueError("Frozen option outside its bank")
        p = render(d, m, [j])[0]
        measured = prev.metrics(p, d, y, v, cut)
        ok = eq.ceiling.eligible_additive(measured)
        if ok != bool(valid[j]):
            raise ValueError("Qualification disagreement on frozen choice")
        # Retain only standalone-eligible anchors in pair searches.
        pool = safe[keep[k][idx]] if safe is not None else np.empty((0, *fixed.shape))
        if len(pool) and not eq.ceiling.eligible_additive(prev.metrics(pool[0], d, y, v, cut)):
            raise ValueError("Incorrect capacity witness")
        fixed_pair = fixed_ok and witness(d, pool, fixed, y, v, cut, mass)
        own_pair = ok and witness(d, pool, p, y, v, cut, mass)
        flexible = fixed_pair if k in ("base", "union") else False
        flexible |= own_pair or any(witness(d, pool, a, y, v, cut, mass) for a in pool)
        out[k] = dict(options=int(keep[k].sum()), any_eligible=int(bool(len(pool))), any_pair=int(flexible),
                      fixed_pair=int(fixed_pair), selected_eligible=int(ok), selected_pair=int(own_pair),
                      selected_easy_unsupported=int(measured["risk"]["completion_easy_risk_upper"] is None))
    if (out["base"]["selected_eligible"] != int(fixed_ok)
            or any(out["union"][key] < out[k][key] for k in MODES[:-1] for key in ("any_pair", "any_eligible", "fixed_pair"))):
        raise ValueError("Union lost a constituent opportunity")
    return out


def score(key, data, source, outer, control, parent, report):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or seal["held_labels_used"] or set(seal["folds"]) != set(parent["folds"]):
        raise ValueError("Require all temporal choices to be frozen before scoring")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    total, city, rows = {k: Counter() for k in MODES}, {}, []
    for name, f in seal["folds"].items():
        held, seed, cut = (f["identity"][k] for k in ("held", "seed", "cut"))
        bucket = city.setdefault(held, {k: Counter() for k in MODES})
        with (np.load(study.checked(f["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as a):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], raw[k])
                np.testing.assert_array_equal(z[k], a[k])
            if z["registration"].item() != key:
                raise ValueError("Wrong temporal bank freeze")
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                if d["city"] != held:
                    raise ValueError("Wrong held city")
                m, keep = menu(d, seed, cut, raw["raw_eq"][lo:hi])
                if m is None:
                    continue
                if digest(m) != z["digest"][i]:
                    raise ValueError("Causal menu changed after freezing")
                anchor = a["additive"][lo:hi]
                selected = {k: int(z[f"choice_{k}"][i]) for k in MODES}
                stat = counts(d, m, keep, y, v, cut, anchor, selected)
                pred = render(d, m, list(selected.values()))
                for k, path in zip(MODES, pred, strict=True):
                    threshold = f["calibration"][k]["threshold"]
                    active = threshold is not None and z[f"score_{k}"][i] >= threshold
                    stat[k].update(released=int(active), released_eligible=int(active and stat[k]["selected_eligible"]))
                    total[k].update(stat[k])
                    bucket[k].update(stat[k])
                rows.append(dict(clip=str(n), city=held, seed=seed, complete=bool(v.all()), stats=stat,
                    changed={k: bool(not np.array_equal(path, anchor)) for k, path in zip(MODES, pred, strict=True)},
                    trajectory={k: assign.cost.audit.direct_trajectory(path, anchor, d["cv"], y, v)
                                for k, path in zip(MODES, pred, strict=True)}))
        beat("score", fold=name, two_slot_views=len(rows))
    for k, parent_key in (("any_eligible", "any_qualified"), ("any_pair", "any_pair"),
                          ("fixed_pair", "fixed_pair_capacity"), ("selected_eligible", "fixed_eligible")):
        if total["base"][k] != report["counts"][parent_key]:
            raise ValueError(f"Corrected base capacity changed: {k}")
    if len(rows) != report["counts"]["two_slots"]:
        raise ValueError("Two-slot roster changed")
    cal = {k: sum(r["calibration"][k]["accepted"] for r in seal["folds"].values()) for k in MODES}
    out = dict(registration=key, freeze=study.artifact(p), result_source="fresh_run",
        two_slot_views=len(rows), unique_scenarios=len({r["clip"] for r in rows}),
        counts={k: dict(v) for k, v in total.items()}, by_city={k: {m: dict(v) for m, v in x.items()} for k, x in city.items()},
        complete_counts={k: {metric: sum(r["stats"][k][metric] for r in rows if r["complete"])
                            for metric in ("any_eligible", "any_pair", "fixed_pair", "selected_eligible")} for k in MODES},
        calibration_accepted=cal, changed={k: sum(r["changed"][k] for r in rows) for k in MODES},
        delta_vs_base={k: {metric: av.interval(rows, lambda r: r["stats"][k][metric]-r["stats"]["base"][metric])
                          for metric in ("any_eligible", "any_pair", "fixed_pair", "selected_eligible")} for k in MODES[1:]},
        forced_trajectory_vs_fixed={k: av.interval(rows, lambda r: r["trajectory"][k]) for k in MODES[1:]},
        no_new_fits=True, pair_capacity_hindsight_only=True, release_scope="standalone reference diagnostic",
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False, original_2pct=True)
    base.immutable(OUT/"summary.json", out)
    print(study.json.dumps({k: out[k] for k in ("two_slot_views", "counts", "calibration_accepted", "changed")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    phase = parser.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, outer, control, parent, report = register()
        if phase == "freeze":
            freeze(key, data, src, roles, outer, control, parent)
        else:
            score(key, data, src, outer, control, parent, report)
        beat(phase+"_complete")


if __name__ == "__main__":
    main()
