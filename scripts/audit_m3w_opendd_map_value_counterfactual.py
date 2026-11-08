"""Isolate test-time map alignment in the frozen map-trained utility head."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_map_path_value as study

run, np, torch, prior, source = study.run, study.np, study.torch, study.prior, study.source


def interval(cells, getter):
    a = np.array([np.mean([getter(row) for row in rows]) for rows in cells.values()])
    draws = np.random.default_rng(817).integers(len(a), size=(3000, len(a)))
    return dict(point=float(a.mean()), interval95=np.quantile(a[draws].mean(1), [.025, .975]).tolist())


@torch.inference_mode()
def evaluate(d, prepared, net, cuts, old):
    bank, cv, on = (prepared[k] for k in ("bank", "cv", "on"))
    x, real, wrong = (torch.from_numpy(prepared[k]) for k in ("x", "real", "wrong"))
    choices, paths = {}, {}
    for k in study.model.MODES:
        p = net(study.design(x, real, wrong, k)).numpy()
        c = np.full(len(cv), study.ANCHOR)
        c[on] = study.model.cost.select(p, study.ANCHOR)["agreement"]
        choices[k] = c
        paths[k] = bank[np.arange(len(cv)), c]
        np.testing.assert_array_equal((paths[k] != cv).any((1, 2)), on)
    other = dict(cv=cv, ols4=prepared["ols"], tangent=prepared["paths"]["tangent"], hindcast=prepared["paths"]["hindcast"])
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    a = prior.score(dict(other, **paths), bank, on, {}, choices, y, valid, rec, cuts)["arms"]
    if a["map"] != old["arms"]["map"]:
        raise ValueError("Map-trained original input did not replay")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    paired = {}
    for k in ("no_map", "wrong_map"):
        changed = choices[k] != choices["map"]
        support = np.zeros(len(cv), dtype=bool)
        support[on] = prepared["exchanged"]
        if np.any(changed & ~support):
            raise ValueError("Perturbation changed a row without map support")
        obs, lower = study.transfer.paired(paths[k], paths["map"], y, valid)
        paired[k] = dict(observed_pp=float(100*obs.sum()/mass), lower_pp=float(100*lower.sum()/mass),
            changed=int(changed.sum()), eligible=int(support.sum()))
    return dict(paired=paired, arms=a, fitted_map_path_choices=old["path_counts"]["map"],
        trained_assignment_margin={k: old["shuffle"]["map"]["direct_lower"]-old["shuffle"][k]["direct_lower"]
                                   for k in ("no_map", "wrong_map")})


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, frozen, _, maps, _ = study.register()
    files = [study.OUT / k for k in ("analysis.json", "head_freeze.json")]
    old, heads = [run.read(p) for p in files]
    if old["registration"] != key or old["heads"] != run.sha(files[1]) or old["confirmation_access"] or heads["held_labels_used_in_fit"]:
        raise ValueError("Require the fully held-location learned heads")
    f = study.OUT / "counterfactual_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=[run.artifact(p) for p in files],
        same_frozen_map_trained_head=True, primary="actual vs simultaneous wrong map, half radius",
        no_training=True, original_2pct=True, confirmation_access=False)))
    nets = study.transfer.frozen_models(*frozen[:5])
    data = {s: study.load_site(s) for s in source.SITES}
    out = {}
    for j, s in enumerate(source.SITES):
        d = data[s]
        cut = np.quantile(np.concatenate([data[k]["cut_samples"] for k in source.SITES if k != s]), [.25, .75])
        raster, _ = study.road.raster_for(s, maps[s])
        rows = []
        for i, seed in enumerate(prior.SEEDS):
            receipt = heads["heads"][f"{s}/{seed}/map"]
            z = torch.load(run.checked(receipt["artifact"]), weights_only=True, map_location="cpu")
            if run.stable(z["identity"]) != receipt["identity"] or z["step"] != study.STEPS:
                raise ValueError("Head does not match held location and seed")
            head = study.model.PathValue()
            head.load_state_dict(z["head"])
            p = study.prepare(d, study.transfer.forecast(d["h"], [nets[seed][j]]), raster, .5)
            rows.append(evaluate(d, p, head.eval(), cut, old["by_site"][s][i]["half"]))
        out[s] = rows
    all_rows = [v for rows in out.values() for v in rows]
    summary = {k: dict(paired={m: interval(out, lambda r: r["paired"][k][m]) for m in ("observed_pp", "lower_pp")},
        changed=sum(r["paired"][k]["changed"] for r in all_rows), eligible=sum(r["paired"][k]["eligible"] for r in all_rows),
        difference_in_training_assignment_margins=interval(out, lambda r: r["trained_assignment_margin"][k]),
        perturbation_passes=sum(r["arms"][k]["pass_views"] for r in all_rows)) for k in ("no_map", "wrong_map")}
    run.once(study.OUT / "counterfactual.json", run.stable(dict(registration=run.sha(f), summary=summary, by_site=out,
        no_training=True, confirmation_access=False)))


if __name__ == "__main__":
    main()
