"""Distinguish an empty predicted risk set from failure of the price search."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dense_motion_calibration as base


np, run, torch = base.np, base.run, base.torch


def feasibility(ref, moments, legal, budget=.02):
    gain, harm, easy = moments.transpose(2, 0, 1)
    r = ref[:, :, None]
    den = np.broadcast_to((r > 0).all(1), gain.shape)
    legal = np.broadcast_to(legal[:, None], gain.shape)
    all_risk = np.divide(harm, ref[:, :1], out=np.full_like(harm, np.inf), where=ref[:, :1] > 0.)
    easy_risk = np.divide(easy, ref[:, 1:], out=np.full_like(easy, np.inf), where=ref[:, 1:] > 0.)
    positive = legal & den & (gain > 0.)
    both = (all_risk <= budget) & (easy_risk <= budget)
    # If every action with a positive easy denominator violates its easy
    # bound, their sums violate it as well. No future label enters this check.
    easy_defined = legal & (ref[:, 1:] > 0.)
    easy_possible = easy_defined & (easy_risk <= budget)
    zero_ref_slack = legal & (ref[:, 1:] == 0.) & (easy == 0.)
    all_possible = legal & (ref[:, :1] > 0.) & (all_risk <= budget)
    min_value = lambda x, m: float(x[m].min()) if m.any() else None
    return dict(
        legal_actions=int(legal.sum()), positive_gain_actions=int(positive.sum()),
        positive_gain_singleton_actions=int((positive & both).sum()),
        easy_within_budget_actions=int(easy_possible.sum()),
        all_within_budget_actions=int(all_possible.sum()),
        min_easy_ratio=min_value(easy_risk, easy_defined),
        min_easy_ratio_on_positive_gain=min_value(easy_risk, positive),
        min_all_ratio_on_positive_gain=min_value(all_risk, positive),
        predicted_easy_infeasible_for_any_nonempty_defined_set=bool(not easy_possible.any()),
        easy_denominator_missing_for_all_actions=bool(not easy_defined.any()),
        zero_easy_reference_zero_harm_actions=int(zero_ref_slack.sum()),
    )


@torch.inference_mode()
def main():
    binding, _, manifest, parent, roster, _, _ = base.register()
    freeze = run.read(base.PUBLIC/"forecast_freeze.json")
    if freeze["registration"] != binding or freeze["calibration_labels_used"]:
        raise ValueError("Require the original label-free freeze")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    nets = base.network_receipt(binding)
    inventory = run.read(base.cal.CACHE)
    meta = {r["sequence_id"]:r for r in inventory["recordings"] if r["sequence_id"] in roster}
    rows = []
    for i,(clip, sel) in enumerate(zip(parent["clips"], freeze["clips"], strict=True), 1):
        rec = clip["sequence"]
        if sel["sequence"] != rec or sel["parent"] != clip["artifact"]:
            raise ValueError("Unexpected frozen forecast provenance")
        f = base.frozen_paths(clip)
        src = base.cal.reader(manifest, rec, meta, inventory["provenance"])
        x, _, bound, _ = base.motion.features(base.history(src, f))
        x, b = torch.from_numpy(x), torch.from_numpy(bound.astype(np.float32))
        legal = np.any(f["cv"] != f["ols"], axis=(1,2))
        for seed, net, prior in zip(base.cal.SEEDS, nets, sel["diagnostics"], strict=True):
            pred = [net(x0, b0) for x0,b0 in zip(x.split(8192), b.split(8192), strict=True)]
            ref, moments = [torch.cat(z).numpy() for z in zip(*pred, strict=True)]
            r = feasibility(ref, moments, legal)
            rows.append(dict(recording=rec, seed=seed, original=prior, **r))
        if i % 10 == 0:
            print(f"Diagnosed {i}/{len(roster)} without labels", flush=True)
    run.once(base.PUBLIC/"predicted_feasibility.json", run.stable(dict(
        rows=rows, registration=binding, future_labels_read=False,
        joint_empty_views=sum(r["original"]["joint_switches"] == 0 for r in rows),
        all_actions_easy_infeasible_views=sum(r["predicted_easy_infeasible_for_any_nonempty_defined_set"] for r in rows),
        positive_gain_singleton_views=sum(r["positive_gain_singleton_actions"] > 0 for r in rows),
    )))


if __name__ == "__main__":
    main()
