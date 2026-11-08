"""Decompose frozen matched-budget utility into observed and missing terms."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import run_m3w_moving_budget_risk as study

run, h, OUT = study.run, study.h, study.OUT
CONTRASTS = ("screen_4_vs_gain", "screen_4_vs_blind", "screen_4_vs_1",
             "screen_8_vs_gain", "screen_8_vs_1")


def main():
    study.torch.set_num_threads(4)
    key = study.register()
    a, h0 = (OUT/f"{k}.json" for k in ("analysis", "head_freeze"))
    previous, heads = run.read(a), run.read(h0)
    if previous["registration"] != key or previous["heads"] != run.sha(h0) or heads["registration"] != key:
        raise ValueError("Require the frozen matched-budget study")
    f = OUT/"attribution_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in (a, h0)],
        contrasts=CONTRASTS, original_2pct=True, confirmation_access=False, no_new_training=True,
        decomposition="paired completion-lower equals observed utility minus unknown disagreement debit")))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    parents, controls, ranks = h.cost.banks()
    cuts = run.cuts_by_source()
    rows = {s: [] for s in previous["by_site"]}
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        _, legal = study.cost_targets(m, y, valid, cuts)
        for s in rows:
            name = f"{m.seed}_{s}"
            net = study.study.trainer.load(heads["heads"][name], lambda: study.model.network(m.x.shape[-1]), study.STEPS)
            z = [study.score(m, rec, ix, groups, y, valid, legal, net, cuts) for rec, (ix, groups) in records.items()
                 if str(d["sites"][ix[0]]) == s]
            effects = {}
            ref = next(r for r in previous["by_site"][s] if r["seed"] == m.seed)
            for k in CONTRASTS:
                x = run.parent.readout.merge_comparisons([r["comparisons"][k] for r in z])
                lower = x["completion_lower_pp"]
                np.testing.assert_allclose(lower, ref["effects"][k], atol=1e-12, rtol=0.)
                effects[k] = dict(observed_pp=x["known_pp"], missing_debit_pp=x["known_pp"]-lower, lower_pp=lower)
            rows[s].append(dict(seed=m.seed, effects=effects))
        print("utility decomposition checked", m.seed, flush=True)
        del m
    draw = np.random.default_rng(73107).integers(len(rows), size=(3000, len(rows)))

    def ci(k, f0):
        z = np.array([np.mean([r["effects"][k][f0] for r in group]) for group in rows.values()])
        return dict(point=float(z.mean()), interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())

    summary = {k: {f0: ci(k, f0) for f0 in ("observed_pp", "missing_debit_pp", "lower_pp")} for k in CONTRASTS}
    run.once(OUT/"attribution.json", run.stable(dict(registration=run.sha(f), by_site=rows, summary=summary,
        original_2pct=True, confirmation_access=False, no_new_training=True)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
