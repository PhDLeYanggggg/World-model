"""Separate a threshold-grid failure from a learned ranking failure on calibration only."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_atomic_pair_admission as pair

np, torch, study, base = pair.np, pair.torch, pair.study, pair.base


def capacity(inputs, truth, cut, predictions):
    opportunities = selected = 0
    for ix, p in pair.menus(inputs):
        supported = False
        for k in range(0, len(p["opts"]), 128):
            paths = pair.paths_for(p, p["opts"][k:k+128])
            if pair.labels(p, paths, truth["y"][ix], truth["v"][ix], cut)[:, 0].any():
                supported = True
                break
        opportunities += int(supported)
        path = predictions["forced_joint"][ix][None]
        selected += int(pair.labels(p, path, truth["y"][ix], truth["v"][ix], cut)[0, 0])
    if selected > opportunities:
        raise ValueError("Selected slate was not contained in the calibration menu")
    return dict(local_strict_available_queries=opportunities, local_strict_selected_queries=selected)


def exact_thresholds(paths, data, cut):
    """Only distinct nonempty score prefixes; cache unchanged recording results."""
    q, p = paths["q"], paths["joint_probability"]
    starts = np.r_[0, np.flatnonzero(np.diff(q))+1]
    score = p[starts]
    valid = np.flatnonzero(score >= 0.)
    order = valid[np.argsort(-score[valid], kind="stable")]
    groups = np.split(order, np.flatnonzero(np.diff(score[order]))+1) if len(order) else []
    current = {k: paths["cv"].copy() for k in ("cal_add", "cal_joint")}
    unique = np.unique(data["rec"])
    ids = {r: np.flatnonzero(data["rec"] == r) for r in unique}
    results = {}
    candidates = []
    stops = np.r_[starts[1:], len(q)]
    for group in groups:
        affected = set()
        for j in group:
            ix = slice(starts[j], stops[j])
            affected.add(data["rec"][starts[j]])
            for arm in ("add", "joint"):
                current[f"cal_{arm}"][ix] = paths[f"forced_{arm}"][ix]
        for rec in sorted(affected):
            ix = ids[rec]
            local = {k: data[k][ix] for k in ("cv", "ols", "y", "v", "q", "rec")}
            rows = pair.assess({k: x[ix] for k, x in current.items()}, local, cut)
            results[rec] = list(rows[0]["arms"].values())
        active = [arm for res in results.values() for arm in res if arm["risk"]["selected"]]
        eligible = all(a["risk"]["supported"] and a["gain"] is not None and min(a["gain"]) > 0. for a in active)
        candidates.append(dict(threshold=float(score[group[0]]), eligible=bool(active and eligible),
                               edit_count=sum(a["risk"]["selected"] for a in active)))
    feasible = [c for c in candidates if c["eligible"]]
    return dict(nonempty_breakpoints=len(candidates), feasible_breakpoints=len(feasible),
                max_feasible_edits=max((r["edit_count"] for r in feasible), default=0))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    freeze_path = pair.OUT/"freeze.json"
    reg_path = pair.OUT/"registration.json"
    frozen, reg = map(study.read, (freeze_path, reg_path))
    if frozen["registration"] != base.sha(reg_path) or frozen["held_targets_used"]:
        raise ValueError("Need the parent role-separated run")
    src = pair.run.turn.exp.loader(reg["cohorts"])
    causal = {s: pair.causal(reg["cohorts"][s]) for s in pair.SITES}
    results = {}
    for name, fit in frozen["fits"].items():
        roles = fit["identity"]
        cal = roles["calibration"]
        if cal == roles["held"] or cal in roles["donors"] or roles["held"] in roles["donors"]:
            raise ValueError("Calibration sources entered the fit")
        state = torch.load(study.checked(fit["checkpoint"]), map_location="cpu", weights_only=True)
        if state["identity"] != roles or state["step"] != pair.STEPS:
            raise ValueError("Wrong checkpoint")
        net = pair.model.PairAdmission().eval()
        net.load_state_dict(state["net"])
        predictions = pair.predict(net, causal[cal])
        cut = frozen["calibration"][name]["cut"]
        results[name] = dict(site=cal, **capacity(causal[cal], src[cal], cut[0], predictions),
                             **exact_thresholds(predictions, src[cal], cut))
    out = dict(freeze=study.artifact(freeze_path), source="fresh_run", sites=results,
        roles="post-hoc calibration-only explanation; no new threshold selected or applied to held",
        any_nonempty_feasible=sum(r["feasible_breakpoints"] > 0 for r in results.values()),
        aggregate={k: sum(r[k] for r in results.values()) for k in (
            "nonempty_breakpoints", "feasible_breakpoints", "local_strict_available_queries", "local_strict_selected_queries")},
        reserved_outcomes_opened=False, independent_confirmation=False)
    base.immutable(pair.OUT/"calibration_audit.json", out)
    print(study.json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
