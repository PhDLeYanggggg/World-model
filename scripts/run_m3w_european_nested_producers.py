"""Build calibration-excluded inner producers without opening reserved roles."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_source_forecast as parent
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from src.evaluation.m3w_nested_calibration import inner_halves, role_sets, check_producer_roles
from src.world_model.m3w_european_source_forecast import SourceForecaster, fit_design
from src.world_model.m3w_native_forecast import fit_trial, predict

PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_nested_calibration_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_calibration_v1'
CONFIG = 'configs/m3w_european_nested_calibration_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_nested_producers.py',
    'src/evaluation/m3w_nested_calibration.py', 'tests/test_m3w_nested_calibration.py',
    'outputs/publication_readiness_2026_09/european_nested_calibration_v1/registration.md')


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    parent.assert_identity(identity['parent_identity'])
    for path, sha in identity['bindings'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen nested-producer registration changed: '+path)
    prior = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_risk_v1/analysis.json'
    if digest(prior) != identity['prior_analysis_sha256']:
        raise ValueError('Diagnostic starting point changed')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    preg, manifest, receipts, pid = parent.load_source()
    if (reg['seeds'] != preg['seeds'] or reg['forecast_steps'] != preg['training']['steps']
            or any(reg[k] for k in ('independent_reserved_readout', 'deployment_promotion', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only forecast budget required')
    data = parent.prepare(manifest, receipts, pid)
    counts = {s: int((data['sites'] == s).sum()) for s in pid['folds']}
    halves = inner_halves(pid['folds'], counts, reg['inner_half_salt'])
    controls = {}
    for key, design, ti in parent.trials(preg, data, pid):
        if ti['kind'] != 'single':
            continue
        complete = parent.complete(key, ti, preg)
        receipt = json.loads((parent.PRIVATE/'predictions'/(key+'.json')).read_text())
        if receipt['identity'] != ti or digest(ROOT/receipt['path']) != receipt['sha256']:
            raise ValueError('Final frozen producer predictions changed')
        controls[key] = dict(checkpoint=dict(path=complete['checkpoint'], sha256=complete['checkpoint_sha256']),
            prediction=dict(path=receipt['path'], sha256=receipt['sha256']), training_sites=ti['fit_sites'])
    prior = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_risk_v1'
    for name in ('verification.json', 'checkpoint_replay.json', 'accounting_audit.json'):
        r = json.loads((prior/name).read_text())
        if r['analysis_sha256'] != digest(prior/'analysis.json') or not r.get('all_passed', True):
            raise ValueError('Verified prior result required')
    identity = dict(bindings={p: digest(ROOT/p) for p in FILES}, parent_identity=pid,
        prior_analysis_sha256=digest(prior/'analysis.json'), halves=halves,
        frozen_final_producers=controls, numpy=np.__version__, torch=torch.__version__)
    immutable_json(PRIVATE/'producer_identity.json', identity)
    roles = [dict(fit_fold=f, calibration_fold=c, **role_sets(pid['folds'], f, c))
        for f in range(3) for c in range(3) if f != c]
    immutable_json(PUBLIC/'producer_matrix.json', dict(identity=identity, roles=roles,
        new_predictors=18, new_predictor_updates=72000, cached_final_predictors=9,
        future_head_fits=54, future_head_updates=108000, source_rows=len(data['sites']),
        result_source='fresh_registration_cached_verified_source_and_final_producers',
        reserved_roles_opened=False))
    return reg, preg, data, identity


def designs(reg, data, identity):
    folds = identity['parent_identity']['folds']
    for fold in range(3):
        parts = identity['halves'][str(fold)]
        for side in range(2):
            design = fit_design(data, parts[side], data['baseline_ade'])
            target = np.flatnonzero(np.isin(data['sites'], parts[1-side]))
            for cal in range(3):
                if cal != fold:
                    roles = role_sets(folds, fold, cal)
                    check_producer_roles(parts[side], parts[1-side], roles['calibration'], roles['readout'])
            for seed in reg['seeds']:
                key = f'fold{fold}_half{side}_seed{seed}'
                binding = dict(identity=identity, fold=fold, half=side, seed=seed,
                    training_sites=parts[side], predicted_sites=parts[1-side],
                    train_ids_sha256=array_hash(design['train_ids']), prediction_ids_sha256=array_hash(target),
                    factors_sha256=array_hash(design['factors']), baseline_index=design['baseline_index'])
                yield key, design, target, binding


def checked(key, binding, steps):
    p = PRIVATE/'producers'/key/'complete.json'
    r = json.loads(p.read_text())
    if (r['identity'] != binding or not r['fit']['complete'] or r['fit']['step'] != steps
            or r['fit']['held_rows_sampled'] or digest(ROOT/r['checkpoint']['path']) != r['checkpoint']['sha256']):
        raise ValueError('Incomplete or changed inner predictor')
    return r


def train(reg, preg, data, identity, resume, pilot):
    for key, design, ids, binding in designs(reg, data, identity):
        directory = PRIVATE/'producers'/key
        if (directory/'complete.json').exists():
            checked(key, binding, reg['forecast_steps'])
            beat('producer_already_complete', trial=key)
            continue
        if shutil.disk_usage(PRIVATE).free < 10*1024**3:
            raise OSError('Below 10GiB free disk; preserve checkpoints and stop new work')
        torch.manual_seed(binding['seed'])
        model = SourceForecaster(preg['architecture'], design['baseline_index'])
        beat('producer_training', trial=key, rows=len(design['train_ids']), prediction_rows=len(ids))
        fit = fit_trial(model, data, design, seed=binding['seed'], settings=preg['training'],
            identity=binding, directory=directory, resume=resume, stop_at=100 if pilot else None,
            heartbeat=lambda **kw: beat(trial=key, **kw))
        cp = directory/'checkpoint.pt'
        if pilot:
            immutable_json(PRIVATE/'pilot.json', dict(identity=binding, fit=fit, checkpoint=artifact(cp),
                resumes_inside_registered_budget=True))
            beat('producer_pilot_complete', trial=key, step=fit['step'])
            return
        state = torch.load(cp, map_location='cpu', weights_only=False)
        unknown = ~data['valid'].any(1)
        if np.any(state['draws'][design['held_ids']]):
            raise ValueError('Excluded locality entered supervised fitting')
        immutable_json(directory/'complete.json', dict(identity=binding, fit=fit, checkpoint=artifact(cp),
            unknown_zero_loss_draws=int(state['draws'][unknown].sum()), result_source='fresh_run'))
        assert_identity(identity)
        beat('producer_complete', trial=key, seconds=fit['seconds'])


def produce(reg, preg, data, identity, verify):
    # Every fitting endpoint must exist before any comparative error readout.
    jobs = list(designs(reg, data, identity))
    for key, _, _, binding in jobs:
        checked(key, binding, reg['forecast_steps'])
    report = []
    for key, design, ids, binding in jobs:
        r = checked(key, binding, reg['forecast_steps'])
        directory = PRIVATE/'producers'/key
        path = directory/'opposite_half.npz'
        receipt = path.with_suffix('.json')
        torch.manual_seed(binding['seed'])
        model = SourceForecaster(preg['architecture'], design['baseline_index'])
        cp = torch.load(ROOT/r['checkpoint']['path'], map_location='cpu', weights_only=False)
        if cp['identity'] != binding or cp['step'] != reg['forecast_steps']:
            raise ValueError('Checkpoint binding changed')
        model.load_state_dict(cp['model'])
        if receipt.exists():
            rr = json.loads(receipt.read_text())
            if rr['identity'] != binding or digest(path) != rr['sha256'] or rr['checkpoint'] != r['checkpoint']:
                raise ValueError('Changed OOF prediction')
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z['ids'])
                if verify:
                    sampled = ids[:4096]
                    fresh = predict(model, data, sampled, 128)
                    np.testing.assert_array_equal(fresh, z['prediction'][:len(sampled)])
            report.append(dict(trial=key, prediction_rows=len(ids), replay_rows=min(4096, len(ids)) if verify else 0,
                checkpoint=r['checkpoint'], prediction=artifact(path), exact=verify))
            beat('producer_prediction_verified', trial=key, replay=verify)
            continue
        if verify:
            raise ValueError('Cannot verify absent prediction bank')
        beat('opposite_half_inference', trial=key, rows=len(ids))
        p = predict(model, data, ids, 128)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, ids=ids, prediction=p)
        os.replace(temporary, path)
        immutable_json(receipt, dict(identity=binding, checkpoint=r['checkpoint'], **artifact(path),
            inference_fields='past_geometry_only_no_target_or_valid_input'))
        report.append(dict(trial=key, prediction_rows=len(ids), checkpoint=r['checkpoint'],
            prediction=artifact(path), exact=False, replay_rows=0))
    assert_identity(identity)
    name = 'producer_replay.json' if verify else 'producer_completion.json'
    immutable_json(PUBLIC/name, dict(identity=identity, checks=report, result_source='fresh_run_checkpoint_replay' if verify else 'fresh_run',
        all_passed=True, predictors=18, updates=72000, calibration_or_outer_error_readout=False))
    beat('producer_bank_complete', replay=verify, predictors=18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'pilot', 'train', 'resume', 'predict', 'replay'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.pilot, args.train, args.predict, args.replay)):
        parser.error('An explicit phase is required')
    if args.pilot and any((args.train, args.predict, args.replay)):
        parser.error('Pilot must be a separate resumable phase')
    for path in (PRIVATE, PUBLIC):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlinked experiment path')
        path.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'producer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        reg, preg, data, identity = load()
        beat('source_verified', rows=len(data['sites']), architecture=platform.machine(), threads=4, workers=0)
        if args.train or args.pilot:
            train(reg, preg, data, identity, args.resume, args.pilot)
        if args.predict:
            produce(reg, preg, data, identity, False)
        if args.replay:
            produce(reg, preg, data, identity, True)
        beat('phase_complete')


if __name__ == '__main__':
    main()
