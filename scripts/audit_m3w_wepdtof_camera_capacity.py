"""Check the action bottleneck on the single exposed WEPDTOF camera."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_readout as fixed
from src.evaluation import m3w_camera_reference_capacity as camera
from src.evaluation import m3w_record_peak_allocation as solver
from src.evaluation.m3w_cv_anchor_budget import candidate_terms

study, base, np, torch = fixed.study, fixed.base, fixed.np, fixed.torch
OUT = study.OUT.parent / 'wepdtof_camera_capacity_v1'
PRIVATE = study.PRIVATE.parent / OUT.name


def pilot():
    roles = study.read(base.OUT/'external_roles.json')
    developed = [k for k, r in roles['groups'].items() if r['role'] == 'development']
    if base.PILOT != 'convenience_store' or developed != [base.PILOT] or not roles['roles_frozen']:
        raise ValueError('Only the exposed pilot may be read')
    parent = base.OUT/'rapid_freeze.json'
    f = study.read(parent)
    if f['annotation_geometry_read'] or f['registration'] != base.sha(study.camera.detector.REG):
        raise ValueError('Require the frozen camera prefix')
    windows = []
    for r in f['windows']:
        p = study.camera.detector.PRIVATE/f"window_{r['start']:06}.json"
        if base.sha(p) != r['cache_sha256']:
            raise ValueError('Changed pilot camera input')
        windows.append(study.read(p))
    return parent, windows


def counts_of(path, cv, q, n):
    rad = np.linalg.norm(path-cv, axis=-1).max(1)
    return np.bincount(q[rad>0.], minlength=n), np.bincount(q, weights=rad, minlength=n)


def finite(cv, ols, bank, y, valid, q, cut, counts, radii):
    weight = 1./np.bincount(q)[q]
    mass, base_gain, gain, _ = candidate_terms(cv, ols, bank, y, valid)
    best = (gain * weight[:, None, None]).max(1).clip(0.)
    relaxed = sum((np.sort(best[q == j], axis=0)[::-1][:n].sum(0)
                   for j, n in enumerate(counts)), start=np.zeros(2))
    upper = 100.*(relaxed + [0., weight@base_gain])/(weight@mass)
    d = solver.design(cv, ols, bank, y, valid, q, cut[0], counts, radii, 'query')
    take, log = solver.solve(d, 10.)
    witness = None
    if take is not None:
        path = cv.copy()
        path[d['row'][take]] = bank[d['row'][take], d['arm'][take]]
        used, peak = counts_of(path, cv, q, len(counts))
        if (used>counts).any() or (peak>radii+1e-8).any() or used.sum()!=take.sum():
            raise ValueError('Solver witness violates matched budget')
        witness = study.metrics(dict(cv=cv, ols4=ols, candidate=path), y, valid,
                                q, base.PILOT, cut, paired_controls=False)['arms']['candidate']
        if min(witness['lower_vs_cv'],witness['lower_vs_ols4']) > 1e-6 and witness['risk']['supported']:
            log['status'] = 'verified'
    return dict(separate_conservative_upper_pp=upper.tolist(), solver=log, witness=witness)


def run(replay):
    parent, windows = pilot()
    reg, cells, _ = study.register()
    cutfile = study.OUT/'pilot_freeze.json'
    pilot_score = study.read(cutfile)
    if (pilot_score['registration'] != reg or pilot_score['external_geometry_read']
            or pilot_score['pilot_camera_sha256'] != base.sha(parent)):
        raise ValueError('Require the same pilot and the original pixel cutoffs')
    paths = [Path(__file__), Path(camera.__file__), Path(solver.__file__), OUT/'protocol.md',
             parent, cutfile, study.OUT/'fixed_pilot_readout.json', study.OUT/'head_freeze.json',
             base.OUT/'external_roles.json']
    registration = OUT/'registration.json'
    base.immutable(registration, dict(inputs=[study.artifact(p) for p in paths],
        study_registration=reg, original_2pct=True, held_scores_or_labels_read=False, source=base.PILOT))
    h, q, ids = study.camera_arrays(windows, pilot=True)
    nets = study.models(reg, cells)
    predictions = study.predict(h, q, nets)
    y, v, match, refs = study.outcomes(windows, h, q, ids, study.references(base.PILOT))
    _, q = np.unique(q, return_inverse=True)
    cut = pilot_score['cut_px']
    prior = study.read(study.OUT/'fixed_pilot_readout.json')['results']
    cells_out = {}
    for s, (net, _) in nets.items():
        z = predictions[s]
        cv, ols = z['cv'], z['ols4']
        old = fixed.method.bank(cv, ols, z['spline'], q)
        x, scale, legal, _ = study.costs.features(h, cv, ols, old)
        with torch.inference_mode():
            est = net(torch.from_numpy(x)).numpy()*scale[:, None, None]
        control = fixed.fixed(est, legal, q, h, cv, old)['fixed_joint']
        eligible = legal.all(1)
        radius = np.where(eligible, np.linalg.norm(old[:, 0]-cv, axis=-1).max(1), 0.)
        counts, radii = counts_of(control, cv, q, int(q.max())+1)
        raw = camera.linear_menu(h, 12)
        extra, possible = camera.matched_menu(cv, raw, eligible, radius)
        rank = camera.past_rank(h, cv, extra, possible, q, counts)
        n, r = counts_of(rank, cv, q, len(counts))
        np.testing.assert_array_equal(n, counts)
        np.testing.assert_allclose(r, radii, rtol=1e-8, atol=1e-8)
        arms = dict(cv=cv, ols4=ols, fixed_joint=control, prefix_rank=rank)
        arms.update(dict(zip(camera.NAMES, raw.swapaxes(0, 1), strict=True)))
        metrics = study.metrics(arms, y, v, q, base.PILOT, cut, paired_controls=False)['arms']
        if metrics['fixed_joint'] != prior[str(s)]['fixed_joint']:
            raise ValueError('Fixed original comparison did not reproduce')
        payload = dict(seed=s, edits=int(counts.sum()), query_total_peak_px=float(radii.sum()),
            eligible=int(eligible.sum()), bounds=camera.radius_bounds(cv, ols, y, v, q, eligible, radius, counts),
            metrics=metrics)
        cp = PRIVATE/f'seed{s}.json'
        identity = dict(registration=base.sha(registration), seed=s)
        if replay and not cp.exists():
            raise ValueError('Seed checkpoint missing for replay')
        if not cp.exists() or replay:
            bank = np.where(eligible[:, None, None, None], old, cv[:, None])
            base.immutable(cp, dict(identity=identity, original=finite(cv, ols, bank, y, v, q, cut, counts, radii),
                expanded=finite(cv, ols, np.concatenate((bank, extra), axis=1), y, v, q, cut, counts, radii)))
        checkpoint = study.read(cp)
        if checkpoint.pop('identity') != identity:
            raise ValueError('Capacity checkpoint input mismatch')
        payload.update(checkpoint)
        cells_out[str(s)] = payload
        print(dict(seed=s, edits=payload['edits'], bound=payload['bounds'],
                   fixed=metrics['fixed_joint']['lower_vs_ols4'], prefix=metrics['prefix_rank']['lower_vs_ols4'],
                   solver={k:checkpoint[k]['solver']['status'] for k in checkpoint}), flush=True)
    base.immutable(OUT/'readout.json', dict(registration=base.sha(registration), cells=cells_out,
        online=len(h), matched=int(match.sum()), complete=int(v.all(1).sum()), reference_histories=refs,
        sources=1, original_2pct=True, held_labels_access=False, independent_confirmation=False,
        new_training=False, deployment_changed=False, submission_ready=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--replay', action='store_true')
    args = cli.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for directory in (OUT, PRIVATE):
        directory.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.replay)
