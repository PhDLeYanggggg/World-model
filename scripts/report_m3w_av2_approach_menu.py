"""Verify paired accuracy and qualified gains after expanding AV2 candidates."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_approach_replay as replay

parent, np, study, base = replay.parent, replay.np, replay.study, replay.base
OUT = replay.OUT


def compare(old_joint, joint, additive, cv, ols, y, valid, cut):
    q = np.zeros(len(y), int)
    a = parent.turn.expanded.actual_contrast(cv, ols, joint, additive, y, valid, q, cut)
    risk = study.policy(joint, cv, cv, y, valid, ["r"]*len(cv),
                        easy_cut=cut[0], hard_cut=cut[1])["recordings"]["r"]
    loss = np.linalg.norm(old_joint-y, axis=-1)-np.linalg.norm(joint-y, axis=-1)
    missing = np.linalg.norm(old_joint-joint, axis=-1)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    known = loss[valid].sum()
    return dict(gain_pp=None if not mass else (100.*np.array([known, known-missing[~valid].sum()])/mass).tolist(),
                qualified_pair_win=bool(a and a["qualified"] and a["pair_native"][1] > 1e-10),
                selected=risk["selected"], risk_pass=risk["supported"],
                undefined_easy=risk["completion_easy_risk_upper"] is None,
                signed_easy_over_2=(risk["known_easy_net_degradation"] or 0.) > .02)


def main():
    oracle_path, replay_path = OUT/"readout.json", OUT/"replay_readout.json"
    oracle, result = map(study.read, (oracle_path, replay_path))
    if (oracle["registration"] != base.sha(OUT/"registration.json")
            or result["registration"] != base.sha(OUT/"replay_registration.json")):
        raise ValueError("Readout lineage changed")
    parent_result = study.read(parent.OUT/"readout.json")
    old = study.read(study.checked(parent_result["freeze"]))
    new = study.read(study.checked(result["freeze"]))
    manifest = study.read(parent.SOURCE/"manifest.json")
    if new["outcomes_used"] or not new["all_parent_predictions_reproduced"] or set(new["banks"]) != set(old["files"]):
        raise ValueError("Incomparable frozen forecasts")
    data, rows = {}, []
    for name, item in old["files"].items():
        with np.load(study.checked(item), allow_pickle=False) as z:
            y, valid, _ = parent.targets(study.checked(manifest["objects"][name]["local"]), z["ids"], z["h"])
            error = np.linalg.norm(z["cv"]-y, axis=-1).mean(1)[valid.all(1)]
            data[name] = (z["city"].item(), y, valid, error[error > 0.])
    for name, (city, y, valid, _) in data.items():
        cut = np.quantile(np.concatenate([e for c, _, _, e in data.values() if c != city]), [.25, .75])
        with (np.load(study.checked(old["files"][name]), allow_pickle=False) as b,
              np.load(study.checked(new["banks"][name]), allow_pickle=False) as a):
            for seed in parent.slate.SEEDS:
                args = [x[f"{arm}_{seed}"] for x, arm in ((b, "relative_joint"), (a, "relative_joint"), (a, "additive"))]
                v = compare(*args, b["cv"], b["ols4"], y, valid, cut)
                rows.append(dict(city=city, clip=name, seed=seed, complete=bool(valid.all()), **v))
    qualified = [r for r in rows if r["qualified_pair_win"]]
    active = [r for r in rows if r["selected"]]
    if sum(r["risk_pass"] for r in active) != result["results"]["own"]["relative_joint"]["risk_pass"]:
        raise ValueError("Expanded risk passes did not replay")
    new_scenes = {(r["city"], r["clip"]) for r in oracle["rows"]
                  if r["arms"]["own"]["best"] is not None and r["arms"]["base"]["best"] is None}
    support = dict(scored_views=len(rows), scenarios=len(data), risk_pass=sum(r["risk_pass"] for r in active),
        edited_views=len(active), undefined_easy=sum(r["undefined_easy"] for r in active),
        qualified_pair_wins=len(qualified), complete_qualified_pair_wins=sum(r["complete"] for r in qualified),
        unique_qualified_scenarios=len({r["clip"] for r in qualified}),
        added_strict_hindsight_scenarios_by_city=dict(Counter(c for c, _ in new_scenes)))
    summary = dict(source="fresh_run", checkpoints="cached_verified", independent_confirmation=False,
        capacity=oracle["results"], neighbor_supported_views=oracle["neighbors_supported_views"],
        replay=result["results"], support=support,
        direct_gain_own_joint_over_old_joint_pp=parent.interval(rows, lambda r: r["gain_pp"]),
        receipts={k: study.artifact(p) for k, p in (("oracle", oracle_path), ("replay", replay_path))})
    base.immutable(OUT/"summary.json", summary)
    print(study.json.dumps({k: summary[k] for k in ("support", "direct_gain_own_joint_over_old_joint_pp")}, indent=2))


if __name__ == "__main__":
    main()
