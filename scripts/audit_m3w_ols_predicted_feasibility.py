"""Separate search failure from predicted risk infeasibility, without outcomes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_ols_relative_gain as study


np, torch, run, cal, target, model = study.np, study.torch, study.run, study.cal, study.target, study.model


def relaxed_excess(changed, harm, refs):
    excess = np.where(changed[..., None], harm-model.TOLERANCE*refs[:, None], 0.)
    return excess[:, 0].sum(0), excess.min(1).sum(0)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, manifest, old, roster, _ = study.register()
    p = study.PUBLIC/"freeze.json"
    seal = run.read(p)
    if seal["registration"] != key or seal["labels_used"] or seal["confirmation_access"]:
        raise ValueError("Require outcome-free frozen predictions")
    trained = run.read(run.checked(seal["training"]))
    if trained["registration"] != key or trained["calibration_labels_used"]:
        raise ValueError("Incorrect fitted models")
    nets = []
    for f in trained["fits"]:
        ck = torch.load(run.checked(f["checkpoint"]), weights_only=True, map_location="cpu")
        net = model.HarmHead(*ck["identity"]["widths"]).eval()
        net.load_state_dict({k.removeprefix("1."): v for k, v in ck["net"].items() if k.startswith("1.")})
        nets.append(net)
    meta, provenance, _ = target.context(roster)
    bounds, counts = [[] for _ in nets], 0
    for i, (cell, frozen) in enumerate(zip(old["clips"], seal["records"], strict=True), 1):
        rec = cell["sequence"]
        if rec != frozen["recording"]:
            raise ValueError("Wrong calibration recording")
        d, src = target.old_arrays(cell), cal.reader(manifest, rec, meta, provenance)
        np.testing.assert_array_equal(src.query_frames, d["frames"])
        for f, lo, hi in zip(d["frames"], d["offsets"][:-1], d["offsets"][1:], strict=True):
            q = src.get_inputs(int(f))
            at = q["history_complete"]
            np.testing.assert_array_equal(q["agent_ids"][at], d["agents"][lo:hi])
            a, o, bank, scale = model.features(q["positions"][at])
            bank[:, 0], bank[:, 1] = d["cv"][lo:hi], d["ols"][lo:hi]
            changed = (bank[:, 1:] != bank[:, :1]).any((2, 3))
            aa, oo = torch.from_numpy(a), torch.from_numpy(o)
            for net, data in zip(nets, bounds, strict=True):
                v, r = net(aa, oo)
                data.append(relaxed_excess(changed, v.numpy()[..., 1:]*scale[:, None, None], r.numpy()*scale[:, None]))
            counts += 1
        if i % 10 == 0:
            print(f"{i}/{len(roster)} predicted-risk bounds", flush=True)
    by_seed = {}
    for seed, data in zip(study.SEEDS, bounds, strict=True):
        z = np.asarray(data)
        if (z[:, 1] > z[:, 0]+1e-10).any():
            raise ValueError("Relaxation cannot exceed OLS4")
        violations = z[:, 1] > 0.
        by_seed[str(seed)] = dict(relaxed_all_violation=int(violations[:, 0].sum()),
            relaxed_easy_violation=int(violations[:, 1].sum()), relaxed_either_violation=int(violations.any(1).sum()),
            ols_either_violation=int((z[:, 0] > 0.).any(1).sum()),
            median_relaxed_excess=np.median(z[:, 1], axis=0).tolist())
    out = dict(code=run.artifact(Path(__file__)), freeze=run.artifact(p), queries=counts, by_seed=by_seed,
        result_source="fresh_run", target_outcomes_used=False, confirmation_access=False,
        interpretation="Each risk constraint minimized separately over all candidate paths, regardless of predicted gain. "
                       "Positive minimum certifies infeasibility under the model predictions only.")
    run.once(study.PUBLIC/"predicted_feasibility.json", run.stable(out))
    print(by_seed, flush=True)


if __name__ == "__main__":
    main()
