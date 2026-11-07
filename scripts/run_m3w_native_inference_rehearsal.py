"""Rebuild native prefixes and replay frozen comparators on development only."""
import argparse
import os
from pathlib import Path
import platform
import resource
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.audit_m3w_dronecrowd_cross_archive import HOME, digest, once, read, verified
from src.data_unification.m3w_moving_forecast_data import array_hash
from src.world_model.m3w_european_source_forecast import BASELINES, SourceForecaster
from src.world_model.m3w_kinematic_forecast import KinematicForecaster
from src.world_model.m3w_native_prefix_inference import prepare_prefix, predict_relative

PUBLIC = HOME/'dronecrowd_native_inference_rehearsal_v1'
KIN = HOME/'moving_direct_kinematic_forecast_v1'
PRE = HOME/'dronecrowd_blind_preflight_v1'
ARCHIVE = HOME/'dronecrowd_three_seed_bank_v1/analysis.json'
COMBINED = HOME/'dronecrowd_combined_forecaster_coverage_v1/analysis.json'
POPULATION = HOME/'moving_producer_crossfit_v1/population.json'
OLD = HOME/'moving_source_forecast_supported_v1/analysis.json'
SEEDS = {17, 29, 43}
KEYS = ('geometry', 'history', 'origin', 'agents', 'frames', 'queries', 'recordings', 'global_ids')


def build_development(d):
    order = np.argsort(d['queries'], kind='stable')
    ids = d['queries'][order]
    blocks = np.split(order, np.flatnonzero(np.diff(ids))+1)
    g = np.empty_like(d['geometry'])
    cv = np.empty((len(g), 12, 2))
    ols = np.empty_like(cv)
    for ix in blocks:
        q = int(d['frames'][ix[0]])
        if np.any(d['frames'][ix] != q) or np.any(d['recordings'][ix] != d['recordings'][ix[0]]):
            raise ValueError('A query must belong to one recording and frame')
        mask = np.ones((len(ix), 8), bool)
        z = prepare_prefix(dict(positions=d['history'][ix], history_mask=mask,
            history_complete=mask.all(1), observation_frames=np.arange(q-7, q+1),
            query_frame=q, agent_ids=d['agents'][ix]))
        np.testing.assert_array_equal(z.agent_id, d['agents'][ix])
        np.testing.assert_array_equal(z.origin, d['origin'][ix])
        g[ix], cv[ix], ols[ix] = z.geometry, z.baselines[BASELINES[1]], z.baselines[BASELINES[4]]
    np.testing.assert_array_equal(g, d['geometry'])
    np.testing.assert_array_equal(cv.astype(np.float32), g[:, 332:356].reshape(-1, 12, 2))
    return g, dict(queries=len(blocks), rows=len(g), recordings=len(np.unique(d['recordings'])),
        geometry_sha256=array_hash(g), cv_sha256=array_hash(cv), ols4_sha256=array_hash(ols))


def locate(ids, wanted):
    ix = np.searchsorted(ids, wanted)
    if np.any(ix >= len(ids)) or not np.array_equal(ids[ix], wanted):
        raise ValueError('Original forecast rows are not in the development roster')
    return ix


def run():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    pre, coverage = (read(PRE/name) for name in ('analysis.json', 'kinematic_coverage.json'))
    pre_replay = PRE/'replay_execution.json'
    if (read(pre_replay)['analysis_sha256'] != digest(PRE/'analysis.json') or pre['scientific_roles_assigned']
            or pre['future_label_method_called'] or coverage['predictive_admission']):
        raise ValueError('Keep external roles unassigned and outcomes sealed')
    for name, expected in coverage['source_sha256'].items():
        verified(dict(path=name, sha256=expected))
    train, forecast = KIN/'training_freeze.json', KIN/'forecast_freeze.json'
    fit, cache = (read(p) for p in (train, forecast))
    parents = (POPULATION, train, forecast, ARCHIVE, OLD, COMBINED, PRE/'analysis.json',
               pre_replay, PRE/'kinematic_coverage.json', Path(__file__),
               ROOT/'src/world_model/m3w_native_prefix_inference.py')
    refs = read(POPULATION)['files']
    d = {name: np.load(verified(refs[name]), mmap_mode='r', allow_pickle=False) for name in KEYS}
    ids = d['global_ids']
    if len(ids) != 125540 or not (np.diff(ids) > 0).all():
        raise ValueError('Require the original full exposed development roster')
    geometry, roster = build_development(d)
    print(dict(state='geometry_exact', **roster), flush=True)

    self_fits = {f['seed']: f for f in fit['fits'] if f['arm'] == 'self'}
    self_paths = {f['seed']: f for f in cache['forecasts'] if f['arm'] == 'self'}
    allowed = {f['seed']: f['fitted'] for f in coverage['checkpoint_receipts']}
    if set(self_fits) != SEEDS or set(self_paths) != SEEDS or set(allowed) != SEEDS:
        raise ValueError('Require all three original self-kinematic seeds')
    kinematic = []
    for seed, f in sorted(self_fits.items()):
        r = self_paths[seed]
        if r['checkpoint'] != f['checkpoint'] or f['checkpoint']['sha256'] != allowed[seed]:
            raise ValueError('Kinematic checkpoint and source screen do not match')
        state = torch.load(verified(f['checkpoint']), map_location='cpu', weights_only=False)
        if state['step'] != 2400 or state['identity'] != f['identity']:
            raise ValueError('Require final screened kinematic fit')
        net = KinematicForecaster(False)
        net.load_state_dict(state['model'])
        actual = predict_relative(net, geometry, batch_size=256)
        with np.load(verified(r['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            np.testing.assert_array_equal(actual, z['prediction'])
        kinematic.append(dict(seed=seed, rows=len(actual), steps=state['step'],
                              prediction_sha256=array_hash(actual)))
        print(dict(state='kinematic_exact', seed=seed, rows=len(ids)), flush=True)

    trials = read(ARCHIVE)['trials']
    screened = {r['key']: r for r in read(COMBINED)['trials']}
    if (len(trials) != 18 or {r['key'] for r in trials} != set(screened)
            or any(r['still_uncovered'] for r in screened.values())
            or {(r['family'], r['fold'], r['seed']) for r in trials} !=
               {(family, fold, seed) for family in ('pct', 'eqmotion') for fold in range(3) for seed in SEEDS}):
        raise ValueError('Require all 18 source-screened public-family adaptations')
    forecasts = {r['sha256']: r for r in read(OLD)['prediction_receipts']}
    family = []
    for t in trials:
        f = forecasts[t['prediction_sha256']]
        ident = f['identity']
        if (f['checkpoint_sha256'] != t['checkpoint']['sha256']
                or any(t[k] != ident[k] for k in ('family', 'fold', 'seed'))):
            raise ValueError('Comparator checkpoint and cached prediction disagree')
        state = torch.load(verified(t['checkpoint']), map_location='cpu', weights_only=False)
        if state['step'] != 4000 or state['identity'] != ident:
            raise ValueError('Require final screened comparator fit')
        take = (np.asarray(t['original_prediction_batch_starts'])[:, None]+np.arange(128)).ravel()
        with np.load(verified(f), allow_pickle=False) as z:
            wanted, expected = z['ids'][take], z['relative_prediction'][take]
        if np.asarray(state['draws'])[wanted].any():
            raise ValueError('Comparator probe includes sampled rows')
        ix = locate(ids, wanted)
        net = SourceForecaster(ident['architecture'], ident['baseline_index'])
        net.load_state_dict(state['model'])
        actual = predict_relative(net, geometry[ix], batch_size=128)
        np.testing.assert_array_equal(actual, expected)
        family.append(dict(key=t['key'], rows=len(actual), steps=state['step'],
                           prediction_sha256=array_hash(actual)))
        print(dict(state='comparator_exact', key=t['key'], rows=len(actual)), flush=True)

    value = dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in parents},
        development=roster, causal_paths=list(BASELINES), kinematic=kinematic, public_family=family,
        verified_fits=len(kinematic)+len(family), engineering_replay_only=True,
        external_inputs_opened=False, external_inference_run=False, future_labels_opened=False,
        scientific_roles_assigned=False, controller_and_reference_cuts_frozen=False,
        optimizer_updates=0, original_rules_unchanged=True, submission_ready=False,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    out = PUBLIC/'analysis.json'
    once(out, value)
    return dict(seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pid=os.getpid(), architecture=platform.machine(), threads=torch.get_num_threads(),
        workers=0, analysis_sha256=digest(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'replay'))
    phase = parser.parse_args().phase
    if phase == 'replay' and not (PUBLIC/'run_execution.json').is_file():
        raise ValueError('Run the rehearsal before replay')
    receipt = run()
    dest = PUBLIC/f'{phase}_execution.json'
    if dest.is_file():
        if read(dest)['analysis_sha256'] != receipt['analysis_sha256']:
            raise ValueError('Previous execution binds a different rehearsal')
    else:
        once(dest, receipt)
    print(dict(phase=phase, **receipt), flush=True)


if __name__ == '__main__':
    main()
