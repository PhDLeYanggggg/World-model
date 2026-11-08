"""Check whether lower signed-risk MAE also improves the harmful tail."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_factorized_count_risk as study

base, parent, run, torch = study.study, study.parent, study.run, study.torch


def diagnostics(prediction, target):
    pos = target > 0.
    error = np.abs(prediction-target)
    return dict(actions=len(target), positives=int(pos.sum()), error_sum=float(error.sum()),
                positive_error_sum=float(error[pos].sum()), tp=int((pos & (prediction > 0.)).sum()),
                fp=int((~pos & (prediction > 0.)).sum()))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, a, ph, _, old = study.register()
    p, h = (study.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    result, heads = run.read(p), run.read(h)
    if (any(v["registration"] != key for v in (result, heads)) or result["heads"] != run.sha(h)
            or result["confirmation_access"] or not result["original_2pct"]):
        raise ValueError("Require frozen fitted risk comparison")
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    totals, by_site = {}, {}
    for held, d in sites.items():
        bank = parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([v["cut_samples"] for s, v in sites.items() if s != held]), .25)
        by_site[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            rank = base.study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: base.study.prior.model.SwapValue(False), base.study.prior.STEPS)
            m = base.study.menus(d, bank, rank)
            xx, scale, _ = base.features(d, bank, m)
            x = xx["motion"]
            y = base.labels(d, bank, m, cut, scale)
            net = base.study.load(heads["heads"][f"{held}_{seed}"], lambda: study.model.Factorized(x.shape[1]), base.STEPS)
            prev = base.study.load(old["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x.shape[1]), base.STEPS)
            with torch.inference_mode():
                t = torch.from_numpy(x)
                predictions = {"signed": prev(t).numpy(), "factorized": study.model.moments_for_score(net(t)), "zero": np.zeros_like(y)}
            use = m["legal"].copy()
            use[:, 0] = False
            cell_result = {}
            for k, v in predictions.items():
                for name, col in (("all", 2), ("easy", 3)):
                    r = diagnostics(v[..., col][use], y[..., col][use])
                    tag = f"{k}_{name}"
                    cell_result[tag] = r
                    total = totals.setdefault(tag, {f: 0 for f in r})
                    for f, vv in r.items():
                        total[f] += vv
            by_site[held].append(dict(seed=seed, **cell_result))
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(v):
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())

    pass_fraction, effects = [], {}
    for name, rows in result["by_site"].items():
        v = np.mean([100.*(r["matched"]["scores"]["factorized_risk"]["dual_positive"]-r["matched"]["scores"]["signed"]["dual_positive"])
                       /r["matched"]["scores"]["signed"]["views"] for r in rows])
        pass_fraction.append(v)
        effects[name] = float(v)
    counts = {}
    for k, v in totals.items():
        counts[k] = dict(**v, mae=v["error_sum"]/v["actions"], positive_mae=v["positive_error_sum"]/v["positives"],
                         tpr=v["tp"]/v["positives"], fpr=v["fp"]/(v["actions"]-v["positives"]))
    run.once(study.OUT / "risk_readout.json", run.stable(dict(registration=key, analysis=run.artifact(p),
        code=run.artifact(Path(__file__)), pooled_action_diagnostic=counts, by_site=by_site,
        same_count_dual_pass_fraction_difference_pp=ci(np.array(pass_fraction)), per_site_pass_difference_pp=effects,
        action_rows_not_independent=True, no_training=True, original_2pct=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
