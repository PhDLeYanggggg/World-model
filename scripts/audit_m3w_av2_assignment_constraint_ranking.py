"""Diagnose frozen feasibility scores without changing any release rule."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_assignment_constraints as run


np, torch, study = run.np, run.torch, run.study
NAMES = ("support", "risk_given_support", "utility_given_support_risk", "all")


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, _, control, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    result = study.read(path)
    f = study.read(study.checked(result["freeze"]))
    if (f["registration"] != key or f["held_labels_used"] or f["reserved_outcomes_opened"]
            or result["independent_confirmation"] or result["reserved_outcomes_opened"]):
        raise ValueError("Require scored development-only results")
    truth = run.prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows, counts = [], Counter()
    sq, predicted, observed = np.zeros((3, 4), dtype=np.float64)
    supports = np.zeros(4, dtype=np.int64)
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        ck = torch.load(study.checked(z["fit"]["checkpoint"]), map_location="cpu", weights_only=True)
        if ck["identity"] != z["identity"] or ck["step"] != run.STEPS:
            raise ValueError("Wrong learned checkpoint")
        net = run.model.head(z["fit"]["width"])
        net.load_state_dict(ck["net"])
        net.eval()
        parent = study.read(study.checked(z["identity"]["parent"]))
        held, seed, cut = (z["identity"][k] for k in ("held", "seed", "cut"))
        for (n, d, p, opts, q), seal in zip(run.views(data, control["folds"][name], held, seed, parent),
                                            z["rows"], strict=True):
            if seal["clip"] != n or seal["hash"] != q["hash"]:
                raise ValueError("Wrong frozen option set")
            y, v = truth[n]
            c, _, x, _ = run.old.model.inputs(d, p, opts, cut[0])
            prob = net(torch.from_numpy(run.model.features(c, x))).sigmoid().numpy()
            score = prob.prod(1)
            if int((score >= run.model.PROB_CUT).sum()) != seal["proposals"][2][2]:
                raise ValueError("Inconsistent frozen proposal count")
            lab = run.model.labels(d, p, opts, y, v, cut[0])
            ok = lab.all(1)
            order = np.argsort(-score, kind="stable")
            rank = np.flatnonzero(ok[order])
            first = int(rank[0])+1 if len(rank) else None
            rows.append(dict(city=held, clip=str(n), max_score=float(score[order[0]]), first=first,
                             soft=bool(ok[q["choice"][0]])))
            counts["options"] += len(opts)
            counts["exact_options"] += int(ok.sum())
            tests = np.column_stack((lab, ok))
            p4 = np.column_stack((prob, score))
            masks = np.column_stack((np.ones(len(lab), dtype=bool), lab[:, 0], lab[:, 0] & lab[:, 1], np.ones(len(lab), dtype=bool)))
            supports += masks.sum(0)
            predicted += np.where(masks, p4, 0.).sum(0, dtype=np.float64)
            observed += (tests & masks).sum(0)
            sq += np.where(masks, (p4-tests)**2, 0.).sum(0, dtype=np.float64)
    if len(rows) != result["views"] or sum(r["soft"] for r in rows) != result["arms"]["soft"]["eligible"]:
        raise ValueError("Incorrect development roster")
    opportunity = [r for r in rows if r["first"] is not None]
    paired = run.av.interval(rows, lambda r: int(r["first"] == 1)-int(r["soft"]))
    summary = dict(result_source="fresh_run", used_for_selection=False, independent_confirmation=False,
        outcome_only=True, reserved_outcomes_opened=False, source=study.artifact(path), views=len(rows),
        options=counts["options"], exact_options=counts["exact_options"], exact_opportunity_views=len(opportunity),
        topk_eligible={str(k): sum(r["first"] <= k for r in opportunity) for k in (1, 3, 5, 10)},
        frozen_soft_top1=sum(r["soft"] for r in rows), unconstrained_rank_top1_minus_soft=paired,
        best_score_quantiles=dict(zip(("min", "median", "p90", "max"),
            np.quantile([r["max_score"] for r in rows], [0., .5, .9, 1.]).tolist(), strict=True)),
        conditional_scores={k: dict(options=int(n), observed_rate=float(o/n), average_score=float(p/n), brier=float(b/n))
                            for k, n, o, p, b in zip(NAMES, supports, observed, predicted, sq, strict=True)})
    run.base.immutable(run.OUT/"frozen_ranking_diagnostic.json", summary)
    print(study.json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
