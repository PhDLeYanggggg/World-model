"""Revalidate saved three-seed comparators without opening external outcomes."""
import hashlib
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '4'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m3w_dronecrowd_cross_archive import HOME, ROOT, digest, once, read, verified
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import predict
import numpy as np
import torch

OLD = HOME/'moving_source_forecast_supported_v1'
PUBLIC = HOME/'dronecrowd_three_seed_bank_v1'
SCOPE = HOME/'dronecrowd_cross_archive_v1'


def fit_exposure(recordings, history_counts, fit_groups, screened):
    groups = set(fit_groups)
    contributing = {node: history_counts[r['clip']] for node, r in recordings.items()
        if r['exclusion_group'] in groups and history_counts[r['clip']] > 0}
    return dict(recordings=sorted(contributing), rows=sum(contributing.values()),
        groups=sorted(groups), unscreened=sorted(set(contributing)-screened))


def existing_batch(ids, available, batch=128, count=8):
    """Keep original contiguous batch boundaries and never select using outcomes."""
    n = len(ids)//batch*batch
    loc = np.searchsorted(available, ids[:n]); matched = loc < len(available)
    matched[matched] &= available[loc[matched]] == ids[:n][matched]
    full = matched.reshape(-1, batch).all(1)
    blocks = np.flatnonzero(full)[:count]
    if len(blocks) != count: raise ValueError('Not enough exact shared A-only batches')
    take = (blocks[:, None]*batch+np.arange(batch)).ravel()
    return take, loc[take]


def main():
    started = time.monotonic()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    paths = {
        'source_registration': OLD/'registration.json', 'training': OLD/'training.json',
        'source_readout': OLD/'analysis.json', 'job_roster': OLD/'job_roster.json',
        'development_admission': HOME/'moving_development_admission_v1/admission.json',
        'structure': HOME/'movingdronecrowd_intake_v1/analysis.json',
        'image_screen_registration': SCOPE/'registration.json', 'image_screen': SCOPE/'analysis.json',
        'past_A_population': HOME/'moving_producer_crossfit_v1/population.json',
    }
    src = {k:read(p) for k,p in paths.items()}
    training = src['training']; jobs = src['job_roster']['jobs']
    if (len(jobs) != 18 or training['complete_primary_models'] != 18 or training['actual_training_seeds'] != [17,29,43]
            or training['held_group_rows_sampled'] != 0 or training['calibration_opened'] or training['confirmation_opened']):
        raise ValueError('Only the complete source-only trained matrix can be reused')
    if src['image_screen']['registration_sha256'] != digest(paths['image_screen_registration']):
        raise ValueError('Changed cross-archive image screen')
    screen = {r['node'] for r in src['image_screen_registration']['exposed']}
    history = {r['clip']:r['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
               for r in src['structure']['recordings']}
    arrays = {k:np.load(verified(src['past_A_population']['files'][k]), mmap_mode='r', allow_pickle=False)
              for k in ('global_ids', 'geometry')}
    if not (np.diff(arrays['global_ids']) > 0).all(): raise ValueError('Expected original sorted global row IDs')
    checkpoints = {Path(x['path']).parent.name:x for x in training['checkpoints']}
    forecasts = {f'{x["identity"]["family"]}_fold{x["identity"]["fold"]}_seed{x["identity"]["seed"]}': x
                 for x in src['source_readout']['prediction_receipts']}
    rows, times = [], []
    for job in jobs:
        key, ident = job['key'], job['identity']; cp, prediction = checkpoints[key], forecasts[key]
        if (ident['registration_sha256'] != digest(paths['source_registration']) or prediction['identity'] != ident
                or prediction['checkpoint_sha256'] != cp['sha256']):
            raise ValueError('Checkpoint, producer and original cached forecast disagree')
        lineage = fit_exposure(src['development_admission']['recordings'], history, ident['fit_groups'], screen)
        if lineage['rows'] != sum(x['indexed'] for x in ident['normalizers'].values()):
            raise ValueError('Recording-level past support does not reproduce the fitted roster size')
        state = torch.load(verified(cp), map_location='cpu', weights_only=False)
        if state['step'] != 4000 or state['identity'] != ident:
            raise ValueError('A final original checkpoint is required')
        with np.load(verified(prediction), allow_pickle=False) as z:
            ids = z['ids']; take, location = existing_batch(ids, arrays['global_ids'])
            np.testing.assert_array_equal(arrays['global_ids'][location], ids[take])
            if np.asarray(state['draws'])[ids].any(): raise ValueError('Held example sampled by predictor')
            expected = z['relative_prediction'][take]
        model = SourceForecaster(ident['architecture'], ident['baseline_index']); model.load_state_dict(state['model'])
        t = time.monotonic(); actual = predict(model, {'geometry': arrays['geometry'][location]}, np.arange(len(take)), 128)
        duration = time.monotonic()-t
        np.testing.assert_array_equal(actual, expected)
        rows.append(dict(key=key, family=ident['family'], fold=ident['fold'], seed=ident['seed'],
            checkpoint=cp, prediction_sha256=prediction['sha256'], steps=state['step'],
            exact_past_only_probe_rows=len(take), original_prediction_batch_starts=take[::128].tolist(),
            probe_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), lineage=lineage))
        times.append(dict(key=key, family=ident['family'], rows=len(take), seconds=duration))
        print(dict(key=key, rows=len(take), seconds=duration, unscreened=len(lineage['unscreened'])), flush=True)
    analysis = dict(sources={str(p.relative_to(ROOT)):digest(p) for p in paths.values()}, trials=rows,
        verified_models=len(rows), exact_probe_rows=sum(x['exact_past_only_probe_rows'] for x in rows),
        covered_folds=sorted({x['fold'] for x in rows if not x['lineage']['unscreened']}),
        new_optimizer_updates=0, external_data_opened=False, scientific_roles_assigned=False,
        baseline_library_not_full_policy=True, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    dest = PUBLIC/'analysis.json'; once(dest, analysis)
    usage = dict(pid=os.getpid(), seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        configured_cpu_threads=4, configured_interop=1, workers=0, timings=times, analysis_sha256=digest(dest))
    once(PUBLIC/'executions'/f'{usage["pid"]}_{time.time_ns()}.json', usage)
    print(dict(state='complete', analysis_sha256=digest(dest), covered_folds=analysis['covered_folds'],
        exact_probe_rows=analysis['exact_probe_rows'], seconds=usage['seconds']), flush=True)


if __name__ == '__main__':
    main()
