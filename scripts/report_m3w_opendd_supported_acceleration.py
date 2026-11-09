"""Audit evidence prediction separately from trajectory and risk performance."""

from pathlib import Path
import sys

from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_supported_acceleration as run
from src.evaluation.m3w_selection_forecast_readout import errors

np, torch, study, base = run.np, run.torch, run.study, run.base


def accuracy(prob, known):
    pos, neg = int(known.sum()), int((~known).sum())
    rank = rankdata(prob)
    return dict(rows=len(known), prevalence=float(known.mean()),
                auc=float((rank[known].sum() - pos*(pos+1)/2)/(pos*neg)) if pos and neg else None,
                brier=float(np.mean((prob-known)**2)), thresholds={
                    str(t): dict(count=int((prob >= t).sum()),
                                 actual_fraction=float(known[prob >= t].mean()) if (prob >= t).any() else None)
                    for t in (.25, .5)})


@torch.inference_mode()
def run_audit():
    report_path = run.OUT/"readout.json"
    report = study.read(report_path)
    freeze = study.read(study.checked(report["freeze"]))
    reg_path = run.OUT/"registration.json"
    reg = study.read(reg_path)
    key = base.sha(reg_path)
    if freeze["registration"] != key or report["registration"] != key or report["reserved_outcomes_opened"]:
        raise ValueError("Expected original development freeze")
    src = run.turn.exp.loader(reg["cohorts"])
    metrics = {}
    for held in run.SITES:
        d = src[held]
        cv, ols, h, y, v = (d[k] for k in ("cv", "ols", "h", "y", "v"))
        cut = float(np.quantile(np.concatenate([s["e"] for key, s in src.items() if key != held]), .25))
        e, _ = errors(cv, y, v)
        known = v.any(1) & (e > 0.) & (e <= cut)
        x, _, _, _ = run.shared.costs.features(h, cv, ols, cv[:, None])
        x = torch.from_numpy(x[:, 0, :run.model.CONTEXT])
        for seed in run.SEEDS:
            name = f"{held}_{seed}"
            receipt = freeze["fits"][name]
            if set(receipt["identity"]["donors"]) != set(run.SITES)-{held}:
                raise ValueError("Evidence classifier fit included held location")
            state = torch.load(study.checked(receipt["checkpoint"]), map_location="cpu", weights_only=True)
            if state["identity"] != receipt["identity"] or state["step"] != run.STEPS:
                raise ValueError("Incomplete or mismatched support head")
            net = run.model.SupportedCost().eval()
            net.load_state_dict(state["net"])
            p = net.support(x).squeeze(-1).sigmoid().numpy()
            metrics[name] = accuracy(p, known)
    vals = [v for v in metrics.values() if v["auc"] is not None]
    metric_summary = {k: float(np.mean([r[k] for r in vals])) if vals else None for k in ("auc", "brier", "prevalence")}
    return dict(result_source="fresh_run", support_classification=metric_summary, fold_support=metrics,
                report=study.artifact(report_path), totals=report["totals"], counts=report["counts"],
                recording_views=report["recording_views"], unique_recordings=report["unique_recordings"],
                fits=report["fits"], no_new_threshold_selection=True, independent_confirmation=False)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = run_audit()
    base.immutable(run.OUT/"summary.json", out)
    print(study.json.dumps(out["support_classification"], indent=2))
