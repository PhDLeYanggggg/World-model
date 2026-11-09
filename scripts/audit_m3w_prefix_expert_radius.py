"""Match edit size to isolate the path effect of prefix backtest features."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_prefix_expert_mixture as study
from src.evaluation.m3w_dut_matched_motion import matched, peak

np, read, checked, artifact, once, stable = (
    study.np, study.read, study.checked, study.artifact, study.once, study.stable)


def main():
    reg, _ = study.register()
    file = study.OUT / "analysis.json"
    freeze = study.OUT / "freeze.json"
    a, f = read(file), read(freeze)
    if (a["registration"] != reg or a["freeze"] != study.sha(freeze)
            or a["confirmation_access"] or not a["original_2pct"] or not f["before_A_labels"]):
        raise ValueError("Require frozen source-excluded prefixes")
    old, parent = study.prior.bank.old, study.prior.parent
    x = old.prior.arrays(("global_ids", "history", "origin", "queries", "sites", "recordings"))
    labels = old.prior.arrays(("target_eval", "valid"))
    y, valid = labels["target_eval"] - x["origin"][:, None], labels["valid"]
    b = study.model.bank(x["history"])
    cv, ols = b[:, 0], b[:, 1]
    cuts = read(old.PARENTS[4])["cuts"]
    cut = {s: cuts[str(v["job"]["producer_fold"])] for v in read(old.PARENTS[3])["forecasts"]
           for s in v["job"]["donor_sites"]}
    rows, roster = [], {}
    for seed in study.SEEDS:
        pred = {}
        for cell in f["cells"]:
            if cell["seed"] == seed and cell["arm"] in ("prefix", "kinematic"):
                with np.load(checked(cell["forecast"]), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["ids"], x["global_ids"])
                    pred[cell["arm"]] = z["pred"].astype(float)
        controls, on = matched(cv, pred["prefix"], np.ones(len(cv), bool),
                               {"kinematic": pred["kinematic"], "ols4": ols})
        # The common-support mask is based only on past-conditioned forecast directions.
        for rec in np.unique(x["recordings"]):
            i = np.flatnonzero(x["recordings"] == rec)
            key, site = str(rec), str(x["sites"][i[0]])
            roster[key] = site
            q, mask, truth = x["queries"][i], valid[i], y[i]
            paths = {study.CV: cv[i], "prefix": controls["neural"][i],
                     "kinematic": controls["kinematic"][i], "ols4": controls["ols4"][i]}
            r = parent.readout.score_recording(paths, {}, truth, mask, q, key,
                                               easy_cut=cut[site]["easy"], hard_cut=cut[site]["hard"])
            _, inv, count = np.unique(q, return_inverse=True, return_counts=True)
            w = 1. / count[inv]
            full = mask.all(1)
            pairs = {}
            for k in ("kinematic", "ols4"):
                values = study.model.full_masses(paths[k], paths["prefix"], paths[study.CV], truth, mask, w)
                complete = study.model.full_masses(paths[k][full], paths["prefix"][full], paths[study.CV][full],
                                                  truth[full], mask[full], w[full])
                pairs[k] = values
                # Keep the denominator of the whole roster for an additive attribution.
                pairs[k + "_complete_contribution"] = dict(reference=values["reference"],
                    known_delta=complete["known_delta"], unknown_radius=0.)
            risk = {k: v["risk"] for k, v in r["metrics"].items()}
            if any(risk[k]["selected"] != int(on[i].sum()) for k in ("prefix", "kinematic", "ols4")):
                raise ValueError("Matched scorer has different selected support")
            rows.append(dict(r, comparisons=pairs, source=site, seed=seed, matched=int(on[i].sum()),
                raw_nonzero=int((peak(pred["prefix"][i] - cv[i]) > 1e-10).sum())))
    result = study.grouped.grouped_readout(rows, roster, study.SEEDS)
    once(study.OUT / "radius_analysis.json", stable(dict(parent=artifact(file), code=artifact(Path(__file__)),
        summary=result, matched_edits=sum(r["matched"] for r in rows),
        raw_nonzero=sum(r["raw_nonzero"] for r in rows), original_2pct=True, confirmation_access=False,
        same_recipient_and_peak=True)))
    print(result["effects"], flush=True)


if __name__ == "__main__":
    main()
