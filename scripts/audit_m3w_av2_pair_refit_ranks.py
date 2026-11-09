"""Attribute the AV2 refit's failed capture without changing any forecast."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_pair_refit as run

np, torch, study, base = run.np, run.torch, run.study, run.base


@torch.inference_mode()
def audit(net, d, seed, truth, cut):
    a, _, paths, feat = run.candidates(d, seed)
    if feat is None:
        return {}
    y, valid = truth
    _, strict, support = run.label_terms(d, paths, a, y, valid, cut)
    x, b, _ = feat
    output = net(torch.from_numpy(x))
    gain = output[:, 0].tanh().numpy()*b
    strict_logit, support_logit = output[:, 1:].numpy().T
    product = output[:, 1].sigmoid().numpy()*np.maximum(gain, 0.)
    top = int(product.argmax())
    counts = Counter(two_slots=1, active=int(product[top] > 0.), oracle=int(strict.any()),
        support=int(support.any()), strict_options=int(strict.sum()), candidates=len(paths),
        strict_eligible_after_gain_sign=int((strict & (gain > 0.)).any()),
        oracle_without_any_positive_gain=int(strict.any() and not (gain > 0.).any()),
        active_top1_strict=int(product[top] > 0. and strict[top]),
        active_top1_supported=int(product[top] > 0. and support[top]))
    for key, s in (("gain", gain), ("strict", strict_logit), ("support", support_logit), ("product", product)):
        order = np.argsort(-s, kind="stable")
        if key == "product":
            order = order[s[order] > 0.]
        for k in (1, 5, 10):
            counts[f"{key}_top{k}"] += int(strict[order[:k]].any())
    return counts


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, old, source, seal = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    frozen = study.read(study.checked(result["freeze"]))
    if result["registration"] != key or frozen["registration"] != key or frozen["held_labels_used"]:
        raise ValueError("Require the frozen AV2 comparison")
    data = run.read_inputs(old, seal)
    truth = run.load_targets(source, data, data)
    roles = ("cal", "held")
    totals = {r: Counter() for r in roles}
    folds = {}
    for name, z in frozen["folds"].items():
        ck = torch.load(study.checked(z["checkpoint"]), weights_only=True, map_location="cpu")
        if ck["identity"] != z["identity"] or ck["step"] != run.STEPS:
            raise ValueError("Wrong archived fold")
        net = run.network().eval()
        net.load_state_dict(ck["net"])
        ident = z["identity"]
        folds[name] = {}
        for role, city in zip(roles, (ident["calibration"], ident["held"]), strict=True):
            values = Counter()
            for n, d in data.items():
                if d["city"] == city:
                    values.update(audit(net, d, ident["seed"], truth[n], z["cut"]))
            totals[role].update(values)
            folds[name][role] = dict(values)
    if (totals["held"]["oracle"] != result["capacity"]["oracle"]
            or totals["cal"]["active_top1_strict"] != result["calibration"]["cal_strict"]):
        raise ValueError("Oracle or calibrated strict count changed during attribution")
    train = [z["donor_strict"]/z["donor_options"] for z in frozen["folds"].values()]
    out = dict(parent=study.artifact(p), result_source="fresh_run", by_fold=folds,
        totals={k: dict(v) for k, v in totals.items()}, positive_train_fraction_range=[min(train), max(train)],
        diagnostic_only=True, new_training=False, held_used_for_selection=False,
        no_threshold_change=True, independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"rank_attribution.json", out)
    print(study.json.dumps(out["totals"], indent=2))


if __name__ == "__main__":
    main()
