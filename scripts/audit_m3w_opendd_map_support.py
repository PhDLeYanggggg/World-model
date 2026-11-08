"""Preserve map support and coarse causal road state in the shuffle control."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_map_direction as study

run, np, torch, source, prior = study.run, study.np, study.torch, study.source, study.prior


def strata(distance, supported):
    side = distance[:, -1] < 0.
    cross = (distance.min(1) < 0.) & (distance.max(1) > 0.)
    return np.where(supported, 1+side.astype(int)+2*cross.astype(int), 0)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _, maps = study.register()
    p = study.OUT / "analysis.json"
    parent = run.read(p)
    if parent["registration"] != key or parent["confirmation_access"]:
        raise ValueError("Require the registered open development result")
    f = study.OUT / "support_registration.json"
    run.once(f, run.stable(dict(parent=run.artifact(p), code=run.artifact(Path(__file__)),
        primary="same count within recording and map support", secondary="also preserve past side and crossing",
        excludes_future_labels_from_strata=True, confirmation_access=False, original_2pct=True)))
    nets = study.transfer.frozen_models(*frozen[:5])
    results = {}
    for j, site in enumerate(source.SITES):
        with np.load(source.prior.PRIVATE / f"{site}_cohort.npz", allow_pickle=False) as z:
            h, y, rec = z["history"], z["future"], np.repeat(z["recording"], np.diff(z["offsets"]))
        valid = np.isfinite(y).all(2)
        raster, _ = study.raster_for(site, maps[site])
        rows = []
        for i, seed in enumerate(prior.SEEDS):
            cv, _, inc, _, delta = study.transfer.forecast(h, [nets[seed][j]])
            paths, on = study.transfer.matched_paths(cv, inc, delta, .5)
            bank = np.stack([paths[k] for k in prior.ARMS], axis=1)
            choices, support, distance = study.method.choose(h, bank, on, raster, prior.REFERENCE)
            coarse = strata(distance, support)
            groups = {"recording": rec, "map_support": np.char.add(np.char.add(rec, ":"), support.astype(int).astype(str)),
                      "map_state": np.char.add(np.char.add(rec, ":"), coarse.astype(str))}
            record = {m: {k: study.shuffle.cell(cv, bank, on, a, y, valid, ids) for k, a in choices.items()}
                      for m, ids in groups.items()}
            if record["recording"] != parent["by_site"][site][i]["half"]["assignment"]:
                raise ValueError("The unstratified control must reproduce exactly")
            rows.append(record)
        results[site] = rows
    draws = np.random.default_rng(817).integers(7, size=(3000, 7))
    summary = {}
    for m in ("recording", "map_support", "map_state"):
        summary[m] = {}
        for k in study.RULES:
            summary[m][k] = {}
            for f2 in next(iter(results.values()))[0][m][k]:
                values = np.array([np.mean([row[m][k][f2] for row in rows]) for rows in results.values()])
                summary[m][k][f2] = dict(point=float(values.mean()), interval95=np.quantile(
                    values[draws].mean(1), [.025, .975]).tolist())
    run.once(study.OUT / "support.json", run.stable(dict(registration=run.sha(f), summary=summary, by_site=results,
        no_training=True, shuffle_risk_not_certified=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
