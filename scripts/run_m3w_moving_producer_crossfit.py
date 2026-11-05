"""Build honest source-held training donors, without a new outer readout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.intake_m3w_movingdronecrowd import sha, once, atomic
from scripts.run_m3w_moving_source_forecast import same_scientific_checkpoint
from src.data_unification import m3w_producer_crossfit as method
from src.data_unification.m3w_moving_forecast_data import array_hash
from src.world_model.m3w_european_source_forecast import SourceForecaster, fit_design
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_eqmotion_adapter import verified_source

NAME = 'moving_producer_crossfit_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs'/f'm3w_{NAME}.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
         ROOT/'src/data_unification/m3w_producer_crossfit.py',
         ROOT/'tests/test_m3w_producer_crossfit.py')


def read(path):
    return json.loads(Path(path).read_text())


def stable(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def checked(ref):
    p = (ROOT/ref['path']).resolve()
    if not p.is_relative_to(ROOT) or sha(p) != ref['sha256']:
        raise ValueError('Artifact identity changed')
    return p


def register():
    cfg = read(CONFIG)
    parent = read(ROOT/cfg['parent_config'])
    previous = read(ROOT/cfg['parent_registration'])
    registry = read(ROOT/cfg['contexts'])
    for bindings in (previous['bindings'], registry['files']):
        for path, digest in bindings.items():
            if sha(ROOT/path) != digest:
                raise ValueError('Changed parent binding: '+path)
    method.load_admission(ROOT, ROOT/cfg['admission'])
    if sha(ROOT/cfg['admission']) != parent['admission_sha256']:
        raise ValueError('Changed admission')
    jobs = method.plan(registry)
    if (len(jobs) != cfg['models'] or cfg['models'] != 6 or cfg['seed'] != 17
            or parent['training']['steps'] != cfg['steps_per_model'] or cfg['steps_per_model'] != 4000
            or cfg['original_risk_budget'] != .02 or not cfg['final_checkpoint_only']
            or any(cfg[k] for k in ('warm_start', 'identity_certification', 'new_outer_C_H_readout',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep fixed within-A experiment and original rules')
    _, author = verified_source()
    value = dict(config=cfg, training=parent['training'], architecture=parent['families']['eqmotion'],
        jobs=jobs, eligibility=method.eligibility(registry),
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parents=[artifact(ROOT/cfg[k]) for k in ('parent_config', 'parent_registration', 'contexts', 'admission')],
        author_core=author, torch=str(torch.__version__), numpy=np.__version__, machine=platform.machine(),
        new_outer_readout=False, role='producer_A_internal_OOF_training_donors',
        original_rules_unchanged=True, independent_confirmation=False)
    once(PUBLIC/'registration.json', stable(value))
    return cfg, value


def require_committed():
    for p in (*FILES, PUBLIC/'registration.json'):
        content = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        if hashlib.sha256(content).hexdigest() != sha(p):
            raise ValueError('Commit registered definition before real-data work')


def runtime(cfg, phase):
    torch.set_num_threads(cfg['torch_threads'])
    torch.set_num_interop_threads(cfg['interop_threads'])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    start = time.monotonic()
    def beat(**values):
        elapsed = time.monotonic()-start
        size = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if elapsed > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume unchanged experiment after 12h boundary')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes'] or size > cfg['artifact_cap_bytes']:
            raise OSError('Registered disk reserve or private size cap exceeded')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        event = dict(phase=phase, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            seconds=elapsed, private_bytes=size,
            peak_RSS_bytes=usage.ru_maxrss if platform.system() == 'Darwin' else usage.ru_maxrss*1024, **values)
        atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', workers=0,
         torch_threads=torch.get_num_threads(), resource_probing=False)
    return beat


def load_data():
    manifest = read(PUBLIC/'population.json')
    data = {}
    for k, ref in manifest['files'].items():
        data[k] = np.load(checked(ref), mmap_mode='r', allow_pickle=False)
        if array_hash(data[k]) != manifest['arrays'][k]['sha256']:
            raise ValueError('Changed packed data')
    return data


def prepare(cfg, reg, beat):
    if (PUBLIC/'population.json').exists():
        data = load_data()
        beat(state='cache_verified', rows=len(data['sites']))
        return
    allowed = set(s for j in reg['jobs'] for s in j['producer_sites'])
    data, receipt = method.pack(ROOT, cfg['admission'], allowed, beat)
    files = {}
    for k, array in data.items():
        dest = PRIVATE/'arrays'/f'{k}.npy'
        dest.parent.mkdir(parents=True, exist_ok=True)
        pending = dest.with_suffix('.pending')
        with pending.open('wb') as f:
            np.save(f, array, allow_pickle=False)
        pending.replace(dest)
        files[k] = artifact(dest)
    receipt['files'] = files
    once(PUBLIC/'population.json', stable(receipt))
    beat(state='packed_complete', rows=receipt['rows'])


def designs(data, reg):
    result = []
    for job in reg['jobs']:
        d = fit_design(data, job['fit_sites'], data['baseline_ade'])
        donor = np.flatnonzero(np.isin(data['sites'], job['donor_sites']))
        if not len(donor) or np.intersect1d(d['train_ids'], donor).size:
            raise ValueError('Invalid donor scope')
        identity = dict(registration_sha256=sha(PUBLIC/'registration.json'),
            population_sha256=sha(PUBLIC/'population.json'), job=job,
            seed=17, architecture=reg['architecture'], baseline_index=d['baseline_index'],
            train_global_ids_hash=array_hash(data['global_ids'][d['train_ids']]),
            donor_global_ids_hash=array_hash(data['global_ids'][donor]),
            factors_hash=array_hash(d['factors']), normalizers=d['normalizers'])
        result.append((job, d, donor, stable(identity)))
    once(PUBLIC/'job_roster.json', dict(jobs=[j[3] for j in result]))
    return result


def fit(data, entry, reg, directory, beat, resume=False, stop_at=None):
    job, design, donor, identity = entry
    torch.manual_seed(17)
    model = SourceForecaster(reg['architecture'], design['baseline_index'])
    result = fit_trial(model, data, design, seed=17, settings=reg['training'], identity=identity,
        directory=directory, resume=resume, stop_at=stop_at,
        heartbeat=lambda **x: beat(job=job['key'], **x))
    state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    if state['draws'][design['held_ids']].any():
        raise ValueError('Nonfitting source sampled')
    return result, state


def pilot(cfg, reg, data, entries, beat):
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Completed pilot immutable; do not rerun')
    entry = entries[0]
    resumed, direct = PRIVATE/'pilot/resumed', PRIVATE/'pilot/direct'
    if not (resumed/'checkpoint.pt').exists():
        fit(data, entry, reg, resumed, beat, stop_at=cfg['pilot_split'])
    result, a = fit(data, entry, reg, resumed, beat, resume=True, stop_at=cfg['pilot_total'])
    _, b = fit(data, entry, reg, direct, beat, resume=(direct/'checkpoint.pt').exists(), stop_at=cfg['pilot_total'])
    same_scientific_checkpoint(a, b)
    m = SourceForecaster(reg['architecture'], entry[1]['baseline_index'])
    ids = entry[1]['train_ids'][:16]
    m.load_state_dict(a['model'])
    pa = predict(m, data, ids, 16)
    m.load_state_dict(b['model'])
    np.testing.assert_array_equal(pa, predict(m, data, ids, 16))
    once(PUBLIC/'pilot.json', stable(dict(real_Torch_CPU_training=True, pid=os.getpid(),
        resume_exact=True, update=cfg['pilot_total'], new_outer_readout=False,
        seconds=result['seconds'], estimate_six_full_fit_seconds=result['seconds']/cfg['pilot_total']*24000,
        pilot_not_scientific_success=True, prefix=artifact(resumed/'checkpoint.pt'))))


def train(cfg, reg, data, entries, beat, resume):
    pilot_info = read(PUBLIC/'pilot.json')
    for i, entry in enumerate(entries):
        job = entry[0]
        home = PRIVATE/'fits'/job['key']
        receipt = PUBLIC/'fits'/f'{job["key"]}.json'
        if receipt.exists():
            checked(read(receipt)['checkpoint'])
            beat(state='completed_fit_verified', job=job['key'])
            continue
        if i == 0 and not (home/'checkpoint.pt').exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked(pilot_info['prefix']), home/'checkpoint.pt')
        result, state = fit(data, entry, reg, home, beat,
                           resume=resume or i == 0)
        if not result['complete'] or state['step'] != 4000:
            raise ValueError('Full fixed budget not complete')
        once(receipt, stable(dict(result_source='fresh_run', job=job,
            checkpoint=artifact(home/'checkpoint.pt'), result=result,
            identity=entry[3], held_sample_draws=0, pid=os.getpid())))
    beat(state='all_six_fits_complete', total_updates=24000)


def infer(reg, data, entry, checkpoint, beat):
    job, d, ids, identity = entry
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['step'] != 4000:
        raise ValueError('Only registered final models allowed')
    if state['draws'][d['held_ids']].any():
        raise ValueError('Teacher exposure violation')
    model = SourceForecaster(reg['architecture'], d['baseline_index'])
    model.load_state_dict(state['model'])
    batches = []
    for start in range(0, len(ids), 4096):
        batches.append(predict(model, data, ids[start:start+4096], 128))
        beat(state='predicting_A_donors_only', job=job['key'], completed_rows=min(start+4096, len(ids)))
    prediction = np.concatenate(batches)
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite OOF forecast')
    return ids, prediction


def forecast(reg, data, entries, beat, replay=False):
    receipts = [read(PUBLIC/'fits'/f'{j[0]["key"]}.json') for j in entries]
    rows = []
    for entry, receipt in zip(entries, receipts):
        job = entry[0]
        path = PRIVATE/'forecasts'/f'{job["key"]}.npz'
        output = PUBLIC/'forecasts'/f'{job["key"]}.json'
        ids, prediction = infer(reg, data, entry, checked(receipt['checkpoint']), beat)
        if replay:
            checked(read(output)['forecast'])
            with np.load(path, allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached['global_ids'], data['global_ids'][ids])
                np.testing.assert_array_equal(cached['relative_prediction'], prediction)
            continue
        if path.exists() or output.exists():
            if not output.exists():
                raise ValueError('Incomplete forecast artifact; verify before recovery')
            checked(read(output)['forecast'])
            with np.load(path, allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached['relative_prediction'], prediction)
                np.testing.assert_array_equal(cached['global_ids'], data['global_ids'][ids])
            rows.append(read(output))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.pending')
        with pending.open('wb') as f:
            np.savez(f, local_ids=ids, global_ids=data['global_ids'][ids], relative_prediction=prediction)
        pending.replace(path)
        a, fde = method.native_errors(prediction+data['origin'][ids, None], data['target_eval'][ids],
                                      data['valid'][ids], np.ones(len(ids)))
        summaries = []
        for site in job['donor_sites']:
            use = data['sites'][ids] == site
            known, end = use & np.isfinite(a), use & np.isfinite(fde)
            cv = data['baseline_ade'][ids, 1]
            summaries.append(dict(source=site, rows=int(use.sum()), known_rows=int(known.sum()),
                endpoint_rows=int(end.sum()), unknown_rows=int((use & ~np.isfinite(a)).sum()),
                neural_ADE=float(a[known].mean()) if known.any() else None,
                causal_CV_ADE=float(cv[known].mean()) if known.any() else None,
                neural_FDE=float(fde[end].mean()) if end.any() else None,
                causal_CV_FDE=float(data['baseline_fde'][ids[end], 1].mean()) if end.any() else None))
        value = stable(dict(result_source='fresh_run', job=job, rows=len(ids), sources=summaries,
            prediction_hash=array_hash(prediction), global_ids_hash=array_hash(data['global_ids'][ids]),
            forecast=artifact(path), checkpoint=receipt['checkpoint'],
            teacher_exposure_disjoint=True, fitting_donors_not_confirmation=True))
        once(output, value)
        rows.append(value)
    if replay:
        once(PUBLIC/'replay.json', dict(pid=os.getpid(), six_predictions_exact=True,
            optimizer_updates=0, independent_confirmation=False, result_source='cached_verified'))
    else:
        all_ids = np.concatenate([data['global_ids'][e[2]] for e in entries])
        if len(np.unique(all_ids)) != len(data['sites']) or len(all_ids) != len(data['sites']):
            raise ValueError('OOF bank must cover each admitted A row exactly once')
        once(PUBLIC/'analysis.json', stable(dict(result_source='fresh_run', models=6,
            optimizer_updates=24000, rows=len(all_ids), unique_source_groups=len(set(data['sites'])),
            source_counts_by_producer={str(f): len(set(s for j in reg['jobs'] if j['producer_fold']==f for s in j['donor_sites'])) for f in (0, 1)},
            forecasts=rows, teacher_exposure_audit_pass=True,
            donor_bank_complete=True, downstream_risk_lift='not_run', deployment_changed=False,
            original_full24_primary='not_estimable', original_rules_unchanged=True,
            identity_certification=False, new_outer_C_H_readout=False,
            submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    beat(state='replay_complete' if replay else 'donor_bank_complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'prepare', 'pilot', 'train', 'forecast', 'replay'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, reg = register()
        if args.phase == 'register':
            print(json.dumps(dict(registered=True, planned_models=len(reg['jobs']))))
            return
        require_committed()
        beat = runtime(cfg, args.phase)
        if args.phase == 'prepare':
            prepare(cfg, reg, beat)
        else:
            data = load_data()
            entries = designs(data, reg)
            if args.phase == 'pilot':
                pilot(cfg, reg, data, entries, beat)
            elif args.phase == 'train':
                train(cfg, reg, data, entries, beat, args.resume)
            else:
                forecast(reg, data, entries, beat, replay=args.phase == 'replay')
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
