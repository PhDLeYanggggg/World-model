"""Test causal references without opening this rotation's held-city outcomes."""

import argparse
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_av2_assignment_order_capacity as capacity
from scripts import run_m3w_av2_assignment_order as prior
from src.world_model import m3w_av2_causal_reference as model

np, torch, study, base, prev, eq, av = prior.np, prior.torch, prior.study, prior.base, prior.prev, prior.eq, prior.av
old, timing, bank = prior.old, prior.timing, prior.bank
OUT, LOCAL = prior.OUT.parent/"av2_causal_reference_v1", prior.LOCAL.parent/"av2_causal_reference_v1"
ARMS = model.ARMS


def register():
    key, data, src, roles, _, outer, control, _, parent, _ = prior.register()
    path = prior.OUT/"summary.json"
    result = study.read(path)
    f = study.read(study.checked(result["freeze"]))
    if f["registration"] != key or f["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require sealed scored parent experiment")
    p = OUT/"registration.json"
    base.immutable(p, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(path), arms=ARMS, seeds=eq.SEEDS, original_2pct=True,
        scope="existing calibration city per rotation only; no held or reserved labels; no new training",
        reference="same two displacement profiles, same causal predictor, same radius at every step and all ordered recipients",
        primary="paired change in exact matched opportunities and conservative trajectory gain vs frozen soft on the same views",
        warning="an easier reference is not an improvement; report standalone qualification and easy degradation",
        guarded="use robust prefix improvement and supported option only when OLS4 distance does not increase",
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(p), data, src, roles, outer, control, parent


def view_paths(data, names, cf, of, seed, cut):
    action, forecaster = timing.rel.pair.prior.load_action(cf), timing.old.load_outer(of)
    raw, offsets = eq.predict(forecaster, data, names)
    for n, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True):
        d = data[n]
        a = timing.incumbent(d, seed, cut, raw[lo:hi], action)
        if a is None or np.any(a != d["cv"], axis=(1, 2)).sum() != 2:
            continue
        p, _, opts, idx, digest = bank.candidate(d, a)
        yield n, d, p, opts, idx, digest


@torch.inference_mode()
def freeze(key, data, roles, outer, control, parent):
    done = {}
    for name, ref in parent["folds"].items():
        f = study.read(study.checked(ref))
        held, city, donors, seed, cut = (f["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut"))
        role = roles[held]
        if city != role["calibration"] or donors != role["donors"] or city in donors or held in [city, *donors]:
            raise ValueError("Overlapping roles")
        ident = dict(f["identity"], parent=ref, registration=key)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            soft = prior.prior.load_soft(f)
            rows = []
            names = [n for n, d in data.items() if d["city"] == city and d["added"]]
            for n, d, p, opts, original, digest in view_paths(data, names, control["folds"][name], outer["folds"][name], seed, cut):
                k, conf = old.predict(soft, d, p, opts, cut)
                choose = model.select(d, p, opts, cut[0], original, k[0])
                rows.append(dict(clip=str(n), hash=digest, choices=choose, soft_conf=float(conf[0])))
            base.immutable(mark, dict(identity=ident, names=names, rows=rows))
        z = study.read(mark)
        if z["identity"] != ident:
            raise ValueError("Wrong reference freeze")
        done[name] = study.artifact(mark)
        print(dict(phase="freeze", completed=len(done), total=len(parent["folds"])), flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, labels_read=False,
                                          held_labels_used=False, reserved_outcomes_opened=False))


def opportunities(paths, ok, picks, d, y, v):
    """Exact outcome-only comparisons; evaluate every legal target against each reference."""
    cv = d["cv"]
    good = paths[ok]
    if not len(good):
        return [False]*len(picks), False, False
    def can_beat(z):
        known, conservative, pair = capacity.contrasts(good, z, cv, y, v)
        return (bool(((conservative >= -capacity.UTILITY_TOL_PP) & (pair[:, 1] > capacity.PAIR_TOL)).any()),
                bool(((known >= -capacity.UTILITY_TOL_PP) & (pair[:, 0] > capacity.PAIR_TOL)).any()))
    exact = [bool(ok[k] and can_beat(paths[k])[0]) for k in picks]
    available = [can_beat(a) for a in good]
    return exact, any(a for a, _ in available), any(b for _, b in available)


def fold_score(data, truth, f, cf, of):
    held, city, seed, cut = (f["identity"][k] for k in ("held", "calibration", "seed", "cut"))
    if city == held or set(truth) != set(f["names"]) or any(data[n]["city"] != city for n in truth):
        raise ValueError("Calibration-only outcome access")
    rows, calibration = [], []
    for (n, d, p, opts, original, digest), r in zip(view_paths(data, f["names"], cf, of, seed, cut), f["rows"], strict=True):
        if r["clip"] != n or r["hash"] != digest or r["choices"][0] != original:
            raise ValueError("Fixed bank changed")
        y, v = truth[n]
        cv = d["cv"]
        paths = np.stack([bank.model.render(cv, p, a) for a in opts])
        ok, _ = bank.model.eligibility(cv, d["ols4"], p, y, v, cut[0], opts)
        matched, any_ref, known = opportunities(paths, ok, r["choices"], d, y, v)
        soft = paths[r["choices"][1]]
        row = dict(clip=str(n), city=city, seed=seed, complete=bool(v.all()), opportunity=bool(ok.any()),
            any_reference_matched=any_ref, any_reference_known_matched=known, arms={})
        for k, i, a in zip(ARMS, r["choices"], matched, strict=True):
            path = paths[i]
            m = prev.metrics(path, d, y, v, cut)
            if eq.ceiling.eligible_additive(m) != bool(ok[i]):
                raise ValueError("Exact candidate and scalar evaluator disagree")
            risk = m["risk"]
            row["arms"][k] = dict(eligible=bool(ok[i]), matched=a, change=i != r["choices"][1],
                easy_defined=risk["completion_easy_risk_upper"] is not None,
                easy_over=(risk["known_easy_net_degradation"] or 0.) > .02,
                ade=m["ade"], fde=m["fde"], tail=m["tail"], hard=m["hard"], gain=m["gain"],
                vs_soft=prior.prior.trajectory(path, soft, d, y, v),
                pair_vs_soft=float(av.frontier.pair_reduction(path[None], soft, y, v)[0, 1]))
        calibration.append(dict(score=r["soft_conf"], strict=row["arms"]["soft"]["eligible"], add_ok=True))
        rows.append(row)
    check = timing.old.action.prior.calibration(calibration)
    parent = study.read(study.checked(f["identity"]["parent"]))
    if check != parent["rules"]["standalone_additive"]:
        raise ValueError("Soft calibration receipt did not replay")
    return rows


def summarize(rows):
    arms = {}
    for k in ARMS:
        a = [r["arms"][k] for r in rows]
        arms[k] = dict(**{n: sum(z[n] for z in a) for n in ("eligible", "matched", "change", "easy_defined", "easy_over")},
            complete_matched=sum(r["complete"] and r["arms"][k]["matched"] for r in rows),
            worst_p95=max(z["tail"] for z in a if z["tail"] is not None),
            **{n: av.interval(rows, lambda r: r["arms"][k][n]) for n in ("ade", "fde", "hard", "gain", "vs_soft", "pair_vs_soft")},
            **{f"{n}_vs_soft": av.interval(rows, lambda r: int(r["arms"][k][n])-int(r["arms"]["soft"][n]))
               for n in ("eligible", "matched")})
    sites = sorted({r["city"] for r in rows})
    return dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), complete_views=sum(r["complete"] for r in rows),
        **{n: sum(r[n] for r in rows) for n in ("opportunity", "any_reference_matched", "any_reference_known_matched")},
        arms=arms, by_city={s: {k: {n: sum(r["arms"][k][n] for r in rows if r["city"] == s)
            for n in ("eligible", "matched", "change")} for k in ARMS} for s in sites})


def score(key, data, src, outer, control, replay):
    p = OUT/"freeze.json"
    seal = study.read(p)
    if seal["registration"] != key or any(seal[k] for k in ("labels_read", "held_labels_used", "reserved_outcomes_opened")):
        raise ValueError("Require outcome-free reference freeze")
    rows = []
    for name, ref in seal["folds"].items():
        f = study.read(study.checked(ref))
        mark = LOCAL/f"{name}.score.json"
        if not mark.exists():
            if replay:
                raise ValueError("Replay may not read outcomes")
            city = f["identity"]["calibration"]
            names = f["names"]
            if city == f["identity"]["held"] or city in f["identity"]["donors"] or any(data[n]["city"] != city for n in names):
                raise ValueError("Wrong label access")
            truth = prev.load_targets(src, data, names)
            base.immutable(mark, dict(freeze=ref, rows=fold_score(data, truth, f, control["folds"][name], outer["folds"][name])))
        z = study.read(mark)
        if z["freeze"] != ref:
            raise ValueError("Wrong score receipt")
        rows.extend(z["rows"])
    s = summarize(rows)
    base.immutable(OUT/"summary.json", dict(**s, freeze=study.artifact(p), result_source="fresh_run", calibration_only=True,
        original_2pct=True, independent_confirmation=False, held_labels_opened=False, reserved_outcomes_opened=False,
        deployment_changed=False, no_new_training=True))
    print(study.json.dumps(s, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score", "replay"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, src, roles, outer, control, parent = register()
        if phase == "freeze":
            freeze(key, data, roles, outer, control, parent)
        else:
            score(key, data, src, outer, control, replay=phase == "replay")


if __name__ == "__main__":
    main()
