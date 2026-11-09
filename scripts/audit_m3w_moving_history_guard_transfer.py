"""Explain zero releases using frozen predictions and previously scored moments."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_history_guard_transfer as stage


np, run = stage.np, stage.run


def main():
    key, _, _, _, records = stage.register()
    p = stage.PUBLIC/"analysis.json"
    result = run.read(p)
    freeze = run.read(run.checked(result["freeze"]))
    if freeze["registration"] != key or result["confirmation_access"] or freeze["moving_labels_used"]:
        raise ValueError("Require the completed development transfer")
    rows = []
    for i, r in enumerate(freeze["records"], 1):
        s = run.read(stage.PRIVATE/f"{i:02d}.score.json")
        if s["freeze"] != r:
            raise ValueError("Incorrect frozen score")
        rows.append(s["scores"])
    independent = [v for z in rows for v in z["independent"]]
    actual = np.asarray([m["risk_ratios"] for m in independent], dtype=float)
    valid = np.isfinite(actual)
    over = actual > .02
    risk_point = np.asarray([t for r in freeze["records"] for t in r["risk_point"]], dtype=float)
    gain_lower = np.asarray([t for r in freeze["records"] for t in r["gain_lower"]], dtype=float)
    blockers = dict(views=len(actual), all_undefined=int((~valid[:, 0]).sum()), easy_undefined=int((~valid[:, 1]).sum()),
        all_over_2=int(over[:, 0].sum()), easy_over_2=int(over[:, 1].sum()), both_over_2=int(over.all(1).sum()),
        actual_median_risk_pp=(100.*np.nanmedian(actual, axis=0)).tolist(),
        predicted_risk_point_pass=int((risk_point <= .02).all(1).sum()),
        predicted_gain_lower_positive=int((gain_lower > 0.).sum()))
    groups = np.repeat([r["source"] for r in records], len(stage.prior.SEEDS))
    by_source = {}
    for s in np.unique(groups):
        ix = np.flatnonzero(groups == s)
        a = stage.prior.merge_comparisons([independent[i]["masses"][1] for i in ix])
        by_source[str(s)] = dict(known_pp=a["known_pp"], debit_pp=a["known_pp"]-a["completion_lower_pp"],
            lower_pp=a["completion_lower_pp"], known_positive=int(sum(independent[i]["masses"][1]["known_delta"] > 0. for i in ix)),
            risk_pass=int(sum(independent[i]["risk"] for i in ix)))
    average = {k: float(np.mean([v[k] for v in by_source.values()])) for k in ("known_pp", "debit_pp", "lower_pp")}
    np.testing.assert_allclose(average["lower_pp"], result["menu"]["independent"]["ols4_lower_pp"], atol=1e-12)
    run.once(stage.PUBLIC/"attribution.json", run.stable(dict(source=run.artifact(p), code=run.artifact(Path(__file__)),
        blockers=blockers, mean_utility_terms=average, by_source=by_source,
        original_2pct=True, confirmation_access=False, new_label_reads=False)))
    print(blockers, average, flush=True)


if __name__ == "__main__":
    main()
