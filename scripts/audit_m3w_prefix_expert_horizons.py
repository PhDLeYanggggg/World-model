"""Test whether within-history expert ordering survives longer leads."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_prefix_expert_mixture as study

np = study.np
LEADS = (1, 4, 8, 12)


def average_prefix_scores(h):
    x, _ = study.model.features(h, "prefix")
    suffix = np.sinh(x[:, 15:].astype(float))
    k = len(study.model.EXPERTS)
    begin, mass = 0, np.zeros((len(h), k))
    for steps in (4, 3, 2, 1):
        end = begin + k * steps
        mass += suffix[:, begin:end].reshape(-1, k, steps).sum(2)
        begin = end
    return mass / 10.


def inspect(d, target, weights):
    n, k = d.shape
    i, j = np.triu_indices(k, 1)
    past = d[:, i] - d[:, j]
    future = target[:, i] - target[:, j]
    informative = (abs(past) > 1e-6) & (abs(future) > 1e-6)
    denom = float(weights @ informative.sum(1))
    concordant = informative & (np.sign(past) == np.sign(future))
    return dict(comparable=denom, agreement=float(weights @ concordant.sum(1)) / denom if denom else None,
                row_pairs=int(informative.sum()))


def main():
    reg, _ = study.register()
    file = study.OUT / "analysis.json"
    a = study.read(file)
    if a["registration"] != reg or a["confirmation_access"]:
        raise ValueError("Require the scored exposed development experiment")
    arrays = study.prior.bank.old.prior.arrays
    x = arrays(("history", "origin", "queries", "sites", "recordings"))
    y = arrays(("target_eval", "valid"))
    full = y["valid"].all(1)
    h = x["history"][full]
    label = y["target_eval"][full] - x["origin"][full, None]
    scores = average_prefix_scores(h)
    pred = study.model.bank(h)
    w = np.zeros(len(x["history"]))
    for key in np.unique(x["recordings"]):
        at = np.flatnonzero(x["recordings"] == key)
        _, inv, n = np.unique(x["queries"][at], return_inverse=True, return_counts=True)
        w[at] = 1. / n[inv]
    w = w[full]
    sites = x["sites"][full]
    names = sorted(set(sites))
    selected = scores.argmin(1)
    output = {}
    draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    for lead in LEADS:
        error = np.linalg.norm(pred[:, :, lead - 1] - label[:, None, lead - 1], axis=-1)
        rows = {}
        for s in names:
            i = np.flatnonzero(sites == s)
            cv = w[i] @ error[i, 0]
            improvement = w[i] @ (error[i, 1] - error[i, selected[i]])
            rows[s] = dict(**inspect(scores[i], error[i], w[i]),
                           prefix_winner_vs_ols4_pp=float(100. * improvement / cv))
        metrics = {}
        for f in ("agreement", "prefix_winner_vs_ols4_pp"):
            v = np.array([r[f] for r in rows.values()])
            metrics[f] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
        output[str(lead)] = dict(rows=rows, summary=metrics)
    study.once(study.OUT / "lead_alignment.json", study.stable(dict(
        parent=study.artifact(file), code=study.artifact(Path(__file__)), complete_rows=int(full.sum()),
        diagnostic_only=True, confirmation_access=False, leads=output)))
    print({k: v["summary"] for k, v in output.items()}, flush=True)


if __name__ == "__main__":
    main()
