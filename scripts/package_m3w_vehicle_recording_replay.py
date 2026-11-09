"""Export anonymous scoring components from the frozen OpenDD anchor study."""

import argparse
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.vehicle_recording_replay import verify
from scripts import package_m3w_reviewer_evidence as archive
from scripts import run_m3w_vehicle_anchor_rank as rank
from src.evaluation.m3w_selection_forecast_readout import errors


np, old, base, pair = rank.np, rank.old, rank.base, rank.pair
TEMPLATE = archive.ROOT/'reproducibility/vehicle_recording_replay'
OUT = rank.OUT.parent/'vehicle_recording_replay_v1'
DEFAULT = archive.LOCAL/'vehicle_recording_replay.zip'


def recording_index(names, roster):
    full, inv = np.unique(names, return_inverse=True)
    at = np.searchsorted(full, roster)
    np.testing.assert_array_equal(full[at], roster)
    return inv, at


def risk_masses(pred, cv, y, valid, easy_cut, group, at):
    pe, _ = errors(pred, y, valid)
    be, _ = errors(cv, y, valid)
    known = valid.any(1)
    easy = known & (be > 0.) & (be <= easy_cut)
    disagreement = np.linalg.norm(pred-cv, axis=-1).max(1)
    selected = disagreement > 0.
    if not np.all(np.bincount(group, weights=selected)[at] == 2):
        raise ValueError('Recording does not have exactly two changed paths')
    signed = np.where(known, pe-be, 0.)
    harm = np.maximum(signed, 0.)
    return np.column_stack([
        np.bincount(group, weights=w)[at] for w in (
            np.where(selected & known, be, 0.), np.where(selected & easy, be, 0.),
            np.where(easy, be, 0.), np.where(selected & known, harm, 0.),
            np.where(selected & easy, harm, 0.), np.where(selected & ~known, disagreement, 0.),
            np.where(easy, signed, 0.))])


def trajectory_terms(ref, pred, y, valid, weight, group, at):
    gain, lower = base.parent.paired(ref, pred, y, valid)
    return np.column_stack([np.bincount(group, weights=weight*x.sum(1))[at] for x in (gain, gain-lower)])


def pair_terms(ref, pred, y, valid):
    i, j = np.triu_indices(len(ref), 1)
    mask = valid[i] & valid[j]
    u, v = ref[i]-ref[j], pred[i]-pred[j]
    target = np.where(mask[..., None], y[i]-y[j], 0.)
    observed = np.linalg.norm(u-target, axis=-1)-np.linalg.norm(v-target, axis=-1)
    debit = np.linalg.norm(u-v, axis=-1)
    return [float(np.where(mask, observed, 0.).sum()), float(np.where(mask, 0., debit).sum()), mask.size]


def direction(z, d, cell, ids, site, rule):
    cv, ols = z['cv'], z['ols4']
    y, valid, w = (d[k] for k in ('future', 'valid', 'weight'))
    names, bounds = z['records'], z['bounds']
    group, at = recording_index(d['recording'], names)
    if any(not np.all(group[lo:hi] == k) for (lo, hi), k in zip(bounds, at, strict=True)):
        raise ValueError('Query does not belong to its recording')
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    pred = {a: pair.prediction(z, a) for a in verify.ARMS}
    ref = pred['additive']
    terms = lambda a, b: trajectory_terms(a, b, y, valid, w, group, at).tolist()
    edges = lambda a, b: [pair_terms(a[lo:hi], b[lo:hi], y[lo:hi], valid[lo:hi]) for lo, hi in bounds]
    all_terms, direct = {}, {}
    for a, p in pred.items():
        all_terms[a] = dict(risk=risk_masses(p, cv, y, valid, cell['cut'][0], group, at).tolist(),
                            cv=terms(cv, p), ols=terms(ols, p), add=terms(ref, p), pair=edges(ref, p))
        if a in verify.FITTED:
            all_terms[a]['score'] = [float(s) if np.isfinite(s) else None for s in z[f'{a}_scores']]
    for a in verify.DIRECT:
        direct[a] = list(zip(terms(pred[a], pred['relative']), edges(pred[a], pred['relative']), strict=True))
    return dict(site=base.SITES.index(site), calibration=base.SITES.index(rule['calibration_site']),
                seed=cell['seed'], cv_mass=mass,
                thresholds={a: [rule['arms'][a]['threshold'], rule['arms'][a]['calibration_recordings']]
                            for a in verify.FITTED},
                recordings=[dict(id=ids.setdefault(name, len(ids)),
                                 arms={a: {k: v[i] for k, v in all_terms[a].items()} for a in verify.ARMS},
                                 direct={a: direct[a][i] for a in verify.DIRECT})
                            for i, name in enumerate(names)])


def export():
    key, source, parent, _, _ = rank.register()
    freeze_path = rank.OUT/'freeze.json'
    frozen, score, paired = (old.read(rank.OUT/p) for p in ('freeze.json', 'readout.json', 'paired_readout.json'))
    digest = rank.audit.sha(freeze_path)
    if (frozen['registration'] != key or score['registration'] != key or score['freeze'] != digest
            or paired['freeze'] != digest or not frozen['held_choices_frozen']
            or set(frozen['cells']) != set(parent['cells'])
            or any(v['external_outcomes_read'] or v['independent_confirmation'] or not v['original_2pct_unchanged']
                   for v in (score, paired))):
        raise ValueError('Expected the original development study and unchanged risk')
    data, ids, results = base.parent.load(source), {}, []
    for name, cell in frozen['cells'].items():
        for site, r in cell['packets'].items():
            rule = cell['rules'][site]
            fit = parent['cells'][name]['fit_sites']
            if site == rule['calibration_site'] or {site, rule['calibration_site']} & set(fit):
                raise ValueError('Evaluation roles overlap training')
            with rank.audit.prior.packet(rank.LOCAL, r) as z:
                results.append(direction(z, data[site], cell, ids, site, rule))
        if len(results) % 14 == 0:
            print(f'Exported {len(results)}/126 directions (PID {os.getpid()})', flush=True)
    return (dict(directions=results, source_resamples=np.random.default_rng(2917).integers(
        len(base.SITES), size=(3000, len(base.SITES))).tolist()), score, paired)


def close(a, b):
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    return isclose(a, b, rel_tol=1e-10, abs_tol=1e-11) if isinstance(b, float) else a == b


def build(output):
    rows, score, paired = export()
    files = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    files['evidence.json'] = (json.dumps(rows, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if any(archive.PRIVATE_MARKERS.search(v) for v in files.values()):
        raise ValueError('Private marker in anonymous archive')
    files['manifest.json'] = archive.main.as_json({k: archive.main.digest(v) for k, v in files.items()})
    with TemporaryDirectory(prefix='vehicle-replay-') as tmp:
        root = Path(tmp)
        a, b = root/'a.zip', root/'b.zip'
        archive.encode(a, files)
        archive.encode(b, files)
        payload = a.read_bytes()
        if payload != b.read_bytes():
            raise ValueError('Archive is not deterministic')
        clean = root/'clean'
        with ZipFile(a) as z:
            z.extractall(clean)
        run = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=clean, check=True,
                             capture_output=True, text=True, timeout=45)
    result = json.loads(run.stdout)
    for k in ('totals', 'by_seed', 'intervals'):
        if not close(result[k], score[k]):
            raise ValueError(f'Replay disagrees with the archived {k}')
    for k in ('active', 'relative_minus'):
        if not close(result[k], paired[k]):
            raise ValueError(f'Replay disagrees with the paired {k}')
    result.update(archive_sha256=archive.main.digest(payload), archive_bytes=len(payload),
                  freeze_sha256=score['freeze'], original_2pct_unchanged=True, submission_ready=False,
                  external_outcomes_read=False, independent_confirmation=False, offline_stdlib=True,
                  deterministic_rebuild=True, source='fresh_replay_of_hash_verified_development_packets')
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError('Refusing to overwrite a different archive')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.with_suffix('.pending.zip').open('xb') as f:
            f.write(payload)
        output.with_suffix('.pending.zip').replace(output)
    OUT.mkdir(parents=True, exist_ok=True)
    old.write(OUT/'receipt.json', result)
    print(json.dumps({k: result[k] for k in ('active', 'ranking_diagnostic', 'archive_sha256', 'archive_bytes')},
                     indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    args = p.parse_args()
    rank.torch.set_num_threads(4)
    rank.torch.set_num_interop_threads(1)
    build(args.output)
