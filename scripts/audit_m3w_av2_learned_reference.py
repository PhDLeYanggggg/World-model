"""Replay the learned AV2 control and isolate support and shortlist losses."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_learned_reference as run

prev, np, torch, study, base = run.prev, run.np, run.torch, run.study, run.base


def load(artifact, factory):
    ck = torch.load(study.checked(artifact["checkpoint"]), weights_only=True, map_location="cpu")
    if ck["identity"] != artifact["identity"] or ck["step"] != run.STEPS:
        raise ValueError("Cannot replay a different fold")
    m = factory().eval()
    m.load_state_dict(ck["net"])
    return m


@torch.inference_mode()
def audit(action, joint, d, seed, y, v, cut):
    point, ref, paths, _, order = run.reference(action, d, seed, cut)
    if paths is None:
        return {}, None
    _, strict, supports = prev.label_terms(d, paths, ref, y, v, cut)
    _, _, ab = prev.label_terms(d, np.stack((point, ref)), ref, y, v, cut)
    add = run.capacity.eligible_additive(prev.metrics(ref, d, y, v, cut))
    _, _, p, _, _ = run.predict(action, joint, d, seed, cut)
    _, chosen, _ = prev.label_terms(d, p[None], ref, y, v, cut)
    if chosen[0] and not strict[order].any():
        raise ValueError("Joint selected an action not in the frozen shortlist")
    return dict(two_slots=1, point_easy_support=int(ab[0]), reference_easy_support=int(ab[1]),
        any_easy_support=int(supports.any()), any_strict=int(strict.any()),
        top8_strict=int(strict[order].any()), selected_strict=int(chosen[0]),
        eligible_reference=int(add), eligible_and_any_strict=int(add and strict.any()),
        eligible_and_top8_strict=int(add and strict[order].any()),
        eligible_and_selected_strict=int(add and chosen[0])), (point, ref, p)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, original, source, seal, _ = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    f = study.read(study.checked(result["freeze"]))
    if result["registration"] != key or f["registration"] != key or f["held_labels_used"]:
        raise ValueError("Only audit the completed development run")
    data = prev.read_inputs(original, seal)
    truth = prev.load_targets(source, data, data)
    folds = {}
    roles = ("calibration", "held")
    for name, z in f["folds"].items():
        ident = z["identity"]
        action = load(z["action"], run.Action)
        joint = load(z["pair"], prev.network) if z["pair"] is not None else None
        fold = folds[name] = {}
        for role in roles:
            q = Counter()
            for n, d in data.items():
                if d["city"] != ident[role]:
                    continue
                count, paths = audit(action, joint, d, ident["seed"], *truth[n], z["cut"])
                q.update(count)
                if role == "held" and paths is not None:
                    with np.load(study.checked(z["forecasts"][n]), allow_pickle=False) as pz:
                        for key, path in zip(("point", "support", "joint"), paths, strict=True):
                            np.testing.assert_array_equal(path, pz[key])
            fold[role] = dict(q)
    totals = {r: dict(sum((Counter(fold[r]) for fold in folds.values()), Counter())) for r in roles}
    if (totals["held"].get("eligible_and_any_strict", 0) != result["counts"].get("support_paired_ceiling", 0)
            or totals["held"].get("selected_strict", 0) != result["counts"].get("joint_strict", 0)):
        raise ValueError("Held replay attribution disagrees with scored result")
    base.immutable(run.OUT/"attribution.json", dict(source=study.artifact(p), code=study.artifact(Path(__file__)),
        totals=totals, by_fold=folds, result_source="fresh_run", diagnostics_only=True,
        no_policy_change=True, independent_confirmation=False, reserved_outcomes_opened=False))
    print(study.json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
