"""Attribute missed feasible pair improvements to shortlist or ranking."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_conditional_pair_gain as run

np, torch, study, base = run.np, run.torch, run.study, run.base


@torch.inference_mode()
def inspect(net, inputs, truth, outputs, cut):
    c = Counter()
    for ix, p in run.pair.menus(inputs):
        scores = []
        y, v, ref = truth["y"][ix], truth["v"][ix], outputs["forced_add"][ix]
        labels = []
        for k in range(0, len(p["opts"]), 128):
            opts = p["opts"][k:k+128]
            x, g = run.pair.encode(p, opts)
            scores.extend(net(torch.from_numpy(x), torch.from_numpy(g)).sigmoid().numpy()[:, 1, 0])
            paths = run.pair.paths_for(p, opts)
            labels.extend(run.target.targets(p, paths, ref, y, v, cut)[1])
        shortlist = np.argsort(-np.asarray(scores), kind="stable")[:run.SHORTLIST]
        labels = np.asarray(labels, dtype=bool)
        possible, short = bool(labels.any()), bool(labels[shortlist].any())
        picked = run.target.targets(p, outputs["pair_gain"][ix][None], ref, y, v, cut)[1][0]
        if picked and not short:
            raise ValueError("The frozen candidate did not belong to the expected shortlist")
        c.update(queries=1, any_pair_opportunity=int(possible), shortlist_opportunity=int(short),
                 selected_pair_win=int(picked))
    return {k: c[k] for k in ("queries", "any_pair_opportunity", "shortlist_opportunity", "selected_pair_win")}


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report_path = run.OUT/"summary.json"
    result = study.read(report_path)
    f = study.read(study.checked(result["freeze"]))
    key, cohorts, _ = run.register()
    if f["registration"] != key or f["held_outcomes_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require the frozen development forecasts")
    data = run.pair.run.turn.exp.loader(cohorts)
    inputs = {s: run.pair.causal(cohorts[s]) for s in run.SITES}
    all_ = {}
    for name, item in f["folds"].items():
        identity = item["fit"]["identity"]
        cal = identity["cal"]
        state = torch.load(study.checked(item["feasibility"]), map_location="cpu", weights_only=True)
        ident = state["identity"]
        if ident["cal"] != cal or ident["held"] != identity["held"] or cal in ident["donors"]:
            raise ValueError("Feasibility shortlist had calibration exposure")
        net = run.pair.model.PairAdmission().eval()
        net.load_state_dict(state["net"])
        all_[name] = inspect(net, inputs[cal], data[cal], run.rank.checked_npz(item["cal"]["forecast"]), item["cut"][0])
    total = dict(sum((Counter(x) for x in all_.values()), Counter()))
    captures = result["captures"]["cal"]["pair_gain"]
    if (total["any_pair_opportunity"] != captures["feasible_and_pair"]
            or total["selected_pair_win"] != captures["selected_feasible_and_pair"]):
        raise ValueError("Calibration opportunity counts did not replay")
    base.immutable(run.OUT/"shortlist_audit.json", dict(summary=study.artifact(report_path), result_source="fresh_run",
        by_fold=all_, totals=total, calibration_only=True, original_2pct=True, independent_confirmation=False,
        reserved_outcomes_opened=False))
    print(study.json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
