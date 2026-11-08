"""One fixed return-to-floor diagnostic; no amplitude, threshold or horizon search."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_local_velocity as exp

np, model, run = exp.np, exp.model, exp.run


def return_weight(t):
    u = ((np.asarray(t)-6.)/6.).clip(0., 1.)
    return 1. - u*u*(3. - 2.*u)


def readout(jobs):
    d = exp.parent.data(jobs, lambda **_: None)
    q = exp.parent.old.fitting_layout(jobs[0]["identity"], d)
    _, _, masks, _, _, paths = model.inputs(d["geometry"])
    target, valid = d["target"], d["valid"]
    ref = paths["ols4"]
    near = paths["fixed_neighbor"] - ref
    ego_steps = np.diff(d["geometry"][:, :16].reshape(-1, 8, 2).astype(float), axis=1)
    s = np.sqrt(np.square(ego_steps).sum(2).mean(1)).clip(.05)
    vel = np.diff(d["geometry"][:, 38:166].reshape(-1, 8, 8, 2).astype(float), axis=2)
    innovation = (vel[:, :, -1]-vel[:, :, -4])/3

    def change(mask):
        a = (innovation*mask[..., None]).sum(1)/mask.sum(1).clip(1)[:, None]
        a *= np.minimum(1., .15*s/np.linalg.norm(a, axis=1).clip(1e-8))[:, None]
        return a[:, None] * (.5*np.minimum(np.arange(1., 13.), 3.))[:, None]

    np.testing.assert_allclose(near, change(masks["near"]), atol=1e-6)
    far = change(masks["outer"])
    paired = (np.linalg.norm(near, axis=2).max(1) > 1e-8) & (np.linalg.norm(far, axis=2).max(1) > 1e-8)
    shared = paired[:, None, None]
    decay = return_weight(np.arange(1., 13.))[None, :, None]
    a, b = ref+near*decay, ref+far*decay
    np.testing.assert_array_equal(a[:, -1], ref[:, -1])
    _, idx = np.unique(d["sites"], return_inverse=True)
    w = np.zeros(len(idx))
    for x in q:
        w[x] = 1./len(x)
    mass = lambda v: np.bincount(idx, weights=w*v, minlength=9)
    cv = np.where(valid, np.linalg.norm(paths["cv"]-np.where(valid[..., None], target, 0.),
                                      axis=2), 0.).sum(1)/valid.sum(1).clip(1)
    norm = mass(cv)
    draws = np.random.default_rng(73107).integers(9, size=(3000, 9))
    interval = exp.parent.old.metric.intervals
    entries = {"near_pulse_vs_ols4": (a, ref), "outer_pulse_vs_ols4": (b, ref),
               "near_pulse_vs_fixed_near": (a, paths["fixed_neighbor"]),
               "near_vs_outer_common_nonidentity": (ref+near*decay*shared, ref+far*decay*shared)}
    result = {n: {k: interval((100*mass(v)/norm)[:, None], draws)
                  for k, v in model.scores(left, right, target, valid).items()}
              for n, (left, right) in entries.items()}
    return dict(contrasts=result, rows=len(idx), shared_nonidentity_rows=int(paired.sum()),
                endpoint_identical_to_ols4=True, weights=decay.flatten().tolist(),
                registered_single_pulse=True, development_targets=False, external_access=False,
                independent_confirmation=False, deployment=False)


if __name__ == "__main__":
    _, jobs = exp.register()
    exp.prev.receipt(exp.PUBLIC, "replay")
    run.once(exp.PUBLIC / "pulse_diagnostic.json", run.stable(readout(jobs)))
