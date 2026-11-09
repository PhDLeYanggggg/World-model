"""Calibrate unchanged AV2 position forecasts using causal scores and sealed outcomes."""

import argparse
import fcntl
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_relative_projection as prior
from scripts.run_m3w_av2_assignment_constraints import load_soft
from src.world_model import m3w_av2_assignment_cost as cost
from src.world_model import m3w_av2_position_release as model

np, torch, study, base, av = prior.np, prior.torch, prior.study, prior.base, prior.av
OUT, LOCAL = prior.OUT.parent/"av2_position_release_v1", prior.LOCAL.parent/"av2_position_release_v1"
ARMS = (*model.ARMS, "calibration_selected")


def register():
    key, data, _, _, _, outer, control, _, _, _ = prior.register()
    p = prior.OUT/"summary.json"
    result = study.read(p)
    f = study.read(study.checked(result["freeze"]))
    if f["registration"] != key or f["held_labels_used"] or result["reserved_outcomes_opened"] or result["independent_confirmation"]:
        raise ValueError("Require a sealed development position forecast")
    study.checked(result["calibration_receipt"])
    r = OUT/"registration.json"
    base.immutable(r, dict(code=[study.artifact(Path(x)) for x in (__file__, model.__file__)],
        parent=study.artifact(p), arms=ARMS, seeds=prior.eq.SEEDS, original_2pct=True,
        policy="unchanged position projection on release, unchanged CV elsewhere",
        scores="lower-quartile replay gain, same with original risk proxies and edited-easy support, and frozen donor cost confidence",
        threshold="unchanged largest nonempty all-qualified calibration prefix per rotation",
        selected="choose score with largest strictly qualifying calibration prefix; ties prefer risk replay, then trajectory replay",
        forecasts_frozen=True, no_new_training=True, reads_raw_future=False,
        independent_confirmation=False, reserved_outcomes_opened=False))
    return base.sha(r), data, outer, control, f, result


@torch.inference_mode()
def confidence(net, d, path, ix, cut):
    safe, raw = model.history_scores(d, path, ix, cut[0])
    # Each affected actor's channel holds its fixed projected path. There is no alternative to rank.
    pool = np.repeat(path[:, None], 2, axis=1)
    c, a, _, _ = cost.inputs(d, pool, ix[None], cut[0])
    _, conf = cost.choose(net.context(torch.from_numpy(c)).numpy(), net.action(torch.from_numpy(a)).numpy(),
                          np.zeros(1), ix[None])
    return dict(zip(model.ARMS, (safe, raw, float(conf[0])), strict=True))


def freeze(key, data, outer, control, parent):
    done = {}
    for name, ref in parent["folds"].items():
        z = study.read(study.checked(ref))
        ident = dict(registration=key, parent=ref)
        mark = LOCAL/f"{name}.json"
        if not mark.exists():
            p = study.read(study.checked(z["identity"]["parent"]))
            cal = study.read(study.checked(p["identity"]["calibration_parent"]))
            soft = study.read(study.checked(cal["identity"]["parent"]))
            if any(soft["identity"][k] != z["identity"][k] for k in ("held", "calibration", "donors", "seed", "cut")):
                raise ValueError("Transferred cost is from the wrong training fold")
            net = load_soft(soft)
            out = {}
            for role in ("calibration", "held"):
                out[role] = []
                for (n, d, _, ix, digest, _), r in zip(prior.inputs(data, p, control["folds"][name], outer["folds"][name], role),
                                                       z["rows"][role], strict=True):
                    if r["clip"] != n or r["hash"] != digest or ix.tolist() != r["edited"]:
                        raise ValueError("Wrong frozen forecast")
                    path = d["cv"].copy()
                    path[ix] = r["paths"][0]
                    out[role].append(dict(clip=str(n), city=d["city"], score=confidence(net, d, path, ix, z["identity"]["cut"])))
            base.immutable(mark, dict(identity=ident, rows=out))
        z = study.read(mark)
        if z["identity"] != ident:
            raise ValueError("Wrong confidence receipt")
        done[name] = study.artifact(mark)
        print(dict(phase="freeze", completed=len(done), total=len(parent["folds"])), flush=True)
    base.immutable(OUT/"freeze.json", dict(registration=key, folds=done, outcomes_read=False, reserved_outcomes_opened=False))


def role_rows(frozen, parent_ref, name, role):
    r = study.read(prior.LOCAL/f"{name}.{role}.score.json")
    if r["freeze"] != parent_ref or r["role"] != role:
        raise ValueError("Wrong frozen outcome receipt")
    for score, result in zip(frozen["rows"][role], r["rows"], strict=True):
        if score["clip"] != result["clip"] or score["city"] != result["city"]:
            raise ValueError("Calibration or held alignment changed")
        yield dict(clip=result["clip"], city=result["city"], seed=result["seed"], complete=result["complete"],
                   result=result["arms"]["frozen_position"], scores=score["score"])


def calibrate(rows):
    return {k: prior.timing.old.action.prior.calibration([
        dict(score=r["scores"][k], strict=r["result"]["eligible"], add_ok=True) for r in rows]) for k in model.ARMS}


def held_metrics(rows, k):
    a = [r for r in rows if r["release"][k]]
    subset = {m: av.interval(a, lambda r: r["result"][m]) if a else None for m in ("gain", "ade", "fde", "hard")}
    return dict(released=len(a), coverage=len(a)/len(rows), edited_tracks=2*len(a),
        released_scenarios=len({r["clip"] for r in a}), complete_releases=sum(r["complete"] for r in a),
        qualified=sum(r["result"]["eligible"] for r in a), risk_fail=sum(not r["result"]["risk"] for r in a),
        easy_over=sum(r["result"]["easy_over"] for r in a),
        floor_fail=sum(r["result"]["gain"] is None or min(r["result"]["gain"]) <= 0. for r in a),
        worst_released_p95=max((r["result"]["tail"] for r in a if r["result"]["tail"] is not None), default=None),
        held_release_precision=sum(r["result"]["eligible"] for r in a)/len(a) if a else None,
        same_release_set=subset,
        full_policy_conservative_cv_gain=av.interval(rows, lambda r: (r["result"]["gain"][0] if r["result"]["gain"] is not None else None)
                                                      if r["release"][k] else 0.),
        by_city={c: dict(released=sum(r["city"] == c for r in a), qualified=sum(r["city"] == c and r["result"]["eligible"] for r in a))
                 for c in sorted({r["city"] for r in rows})})


def score(key, parent, previous):
    p = OUT/"freeze.json"
    f = study.read(p)
    if f["registration"] != key or f["outcomes_read"] or f["reserved_outcomes_opened"] or set(f["folds"]) != set(parent["folds"]):
        raise ValueError("Require outcome-free confidence freeze")
    rules = {}
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        rows = list(role_rows(z, parent["folds"][name], name, "calibration"))
        q = calibrate(rows)
        rules[name] = dict(scores=q, selected=model.choose_calibrated(q), available=sum(r["result"]["eligible"] for r in rows),
                           views=len(rows))
    t = OUT/"calibration.json"
    base.immutable(t, dict(freeze=study.artifact(p), folds=rules, held_receipts_used=False, original_2pct=True))
    rows = []
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        q = rules[name]
        for r in role_rows(z, parent["folds"][name], name, "held"):
            r["release"] = {k: q["scores"][k]["threshold"] is not None and r["scores"][k] >= q["scores"][k]["threshold"]
                            for k in model.ARMS}
            r["release"]["calibration_selected"] = q["selected"] is not None and r["release"][q["selected"]]
            rows.append(r)
    if (len(rows) != previous["held"]["views"] or sum(r["result"]["eligible"] for r in rows) != previous["held"]["arms"]["frozen_position"]["eligible"]
            or sum(r["available"] for r in rules.values()) != previous["calibration"]["arms"]["frozen_position"]["eligible"]):
        raise ValueError("Projection outcome roster did not replay")
    result = dict(views=len(rows), scenarios=len({r["clip"] for r in rows}), calibration_forced_eligible=sum(r["available"] for r in rules.values()),
        held_forced_eligible=sum(r["result"]["eligible"] for r in rows),
        calibration_pass={k: sum(r["scores"][k]["accepted"] for r in rules.values()) for k in model.ARMS},
        calibration_selected={k: sum((r["selected"] or "abstain") == k for r in rules.values()) for k in (*model.ARMS, "abstain")},
        held={k: held_metrics(rows, k) for k in ARMS}, freeze=study.artifact(p), calibration=study.artifact(t),
        result_source="fresh_run", outcome_source="cached_verified", independent_confirmation=False, raw_future_opened=False,
        original_2pct=True, reserved_outcomes_opened=False, deployment_changed=False)
    base.immutable(OUT/"summary.json", result)
    print(study.json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, data, outer, control, parent, previous = register()
        if phase == "freeze":
            freeze(key, data, outer, control, parent)
        else:
            score(key, parent, previous)


if __name__ == "__main__":
    main()
