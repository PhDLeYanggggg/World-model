"""Fixed native-loss EqMotion comparison on approved explored source sites."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 environment required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_native_forecast import (
    load as parent_load, specification, completed, array_hash, assert_current,
    immutable_json, json_write,
)
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_eqmotion_adapter import verified_source

CONFIG = 'configs/m3w_native_eqmotion_v1.json'
CODE = ('scripts/run_m3w_native_eqmotion.py', 'tests/test_m3w_native_eqmotion.py',
        'src/world_model/m3w_eqmotion_adapter.py', 'configs/m3w_eqmotion_source.json')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    parent, data, rows, parent_identity = parent_load()
    if (cfg['sites'] != parent['sites'] or cfg['seeds'] != parent['seeds']
            or cfg['training'] != parent['training'] or any(cfg[k] for k in
                ('threshold_search', 'model_selection', 'independent_confirmation',
                 'risk_calibration', 'closed_role_readout', 'deployment',
                 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Frozen source-only matching required')
    _, author = verified_source()
    bindings = dict(parent_identity['source_bindings'])
    for p in (CONFIG, cfg['registration'], *CODE):
        bindings[p] = file_digest(ROOT/p)
    spec = json.loads((ROOT/'configs/m3w_eqmotion_source.json').read_text())
    for p, digest in author['files_sha256'].items():
        bindings[str(Path(spec['destination'])/p)] = digest
    reference = json.loads((ROOT/parent['reports']/'analysis.json').read_text())
    bindings[str(Path(parent['reports'])/'analysis.json')] = file_digest(ROOT/parent['reports']/'analysis.json')
    views = {}
    for site in cfg['sites']:
        for seed in cfg['seeds']:
            fold, old_id, old_key = specification(parent, data, parent_identity, site, 'native_coordinate', seed)
            path = ROOT/parent['output']/'trials'/old_key/'complete.json'
            record = completed(path, old_id, cfg['training'])
            pr = next(r for r in reference['predictions'] if r['trial'] == old_key)
            for p, sha in ((str(path.relative_to(ROOT)), file_digest(path)),
                           (record['checkpoint'], record['checkpoint_sha256']), (pr['path'], pr['sha256'])):
                if file_digest(ROOT/p) != sha:
                    raise ValueError('Changed reference artifact')
                bindings[p] = sha
            views[f'{site}_seed{seed}'] = dict(site=site, seed=seed, fold=fold,
                reference=record, reference_prediction=pr)
    identity = dict(source_bindings=bindings, config=cfg, author=author,
        population_sha256=parent_identity['population_sha256'],
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine())
    assert_current(identity)
    return cfg, data, rows, views, identity


def trial_identity(identity, key, view):
    fold = view['fold']
    return dict(identity=identity, view=key, held_site=view['site'], seed=view['seed'],
        train_ids_sha256=array_hash(fold['train_ids']), held_ids_sha256=array_hash(fold['held_ids']),
        factors_sha256=array_hash(fold['factors']), normalizers=fold['normalizers'])


def check_matching(cp, reference, fold, steps, batch_size):
    for field in ('train_ids', 'factors', 'draws'):
        np.testing.assert_array_equal(cp[field], reference[field])
    np.testing.assert_array_equal(cp['train_ids'], fold['train_ids'])
    np.testing.assert_array_equal(cp['factors'], fold['factors'])
    if (cp['step'] != steps or reference['step'] != steps
            or cp['draws'].sum() != steps*batch_size
            or cp['draws'][fold['held_ids']].any()):
        raise ValueError('Unmatched budget or held-source training exposure')


def evaluate(cfg, data, rows, views, identity, beat, verify=False):
    root, reports = ROOT/cfg['output'], ROOT/cfg['reports']
    receipts, predictions, archives = {}, {}, []
    for key, view in views.items():
        ti = trial_identity(identity, key, view)
        r = completed(root/'trials'/key/'complete.json', ti, cfg['training'])
        receipts[key] = r
    # Freeze every predictor output before the first new outcome reduction.
    for key, view in views.items():
        r, seed, ids = receipts[key], view['seed'], view['fold']['held_ids']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        ref = torch.load(ROOT/view['reference']['checkpoint'], map_location='cpu', weights_only=False)
        assert cp['identity'] == trial_identity(identity, key, view)
        check_matching(cp, ref, view['fold'], cfg['training']['steps'], cfg['training']['batch_size'])
        torch.manual_seed(seed)
        model = build_forecaster(cfg['architecture'])
        model.load_state_dict(cp['model'])
        beat(state='replaying' if verify else 'predicting', trial=key, rows=len(ids))
        p = predict(model, data, ids)
        path = root/'predictions'/f'{key}.npz'
        if verify and not path.exists():
            raise ValueError('Missing predictions cannot be verified')
        write_arrays(path, dict(ids=ids, prediction=p))
        with np.load(ROOT/view['reference_prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            old = z['prediction'].copy()
        predictions[key] = (p, old)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
    immutable_json(root/'predictions_complete.json', dict(identity=identity, archives=archives,
        predictions_conditioned_on_future_labels=False, all_endpoints_before_readout=True))
    baseline = data['geometry'][:, 332:356].reshape(-1, 12, 2)
    cv, cf = native_errors(baseline, data['target'], data['valid'], data['scale'])
    position = np.repeat(data['geometry'][:, 14:16, None], 12, axis=2).transpose(0, 2, 1)
    ca, ce = native_errors(position, data['target'], data['valid'], data['scale'])
    n = len(cv)
    errors = {name:{s:[np.full(n, np.nan), np.full(n, np.nan)] for s in cfg['seeds']}
              for name in ('eqmotion', 'transformer')}
    masks = dict(complete=data['valid'].all(1), static_history=rows['static_history'],
        moving_history=~rows['static_history'], hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool),
        zero_CV=data['valid'].all(1) & (cv == 0))
    for key, view in views.items():
        fold, seed = view['fold'], view['seed']
        ids, train = fold['held_ids'], fold['train_ids']
        positive = cv[train][np.isfinite(cv[train]) & (cv[train] > 0)]
        easy_cut = float(np.quantile(positive, .25))
        masks['hard'][ids] = cv[ids] >= fold['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= easy_cut)
        for name, p in zip(('eqmotion', 'transformer'), predictions[key]):
            ade, fde = native_errors(p, data['target'][ids], data['valid'][ids], data['scale'][ids])
            errors[name][seed][0][ids], errors[name][seed][1][ids] = ade, fde
    def metric(m, r, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(m[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    def summarize(ade, fde):
        return dict(ADE=metric(ade, cv), FDE=metric(fde, cf),
            zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
            subsets={k:metric(ade, cv, mask) for k, mask in masks.items()})
    summaries = {name:dict(seeds={str(s):summarize(*v) for s,v in group.items()},
        mean_seed=summarize(*np.mean(list(group.values()), axis=0))) for name,group in errors.items()}
    summaries['constant_velocity'] = summarize(cv, cf)
    summaries['constant_position'] = summarize(ca, ce)
    a, b = (summaries[k]['mean_seed']['ADE']['by_scene'] for k in ('eqmotion', 'transformer'))
    result = dict(identity=identity, result_source='fresh_run_12_native_loss_eqmotion_fits_cached_verified_transformer',
        archives=archives, summaries=summaries,
        primary_contrast=paired_scene_contrast([a[s]['gain_percent'] for s in cfg['sites']],
                                             [b[s]['gain_percent'] for s in cfg['sites']]),
        training=[dict(view=k, fit=r['fit'], checkpoint=r['checkpoint'],
                       checkpoint_sha256=r['checkpoint_sha256']) for k,r in receipts.items()],
        parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        matched_samplers=len(views), prediction_rows=sum(len(v['fold']['held_ids']) for v in views.values()),
        annotation_steps_observed=8, annotation_steps_predicted=12, sdd_stride=12,
        family_comparison_not_isolated_equivariance=True, published_best_of_20_reproduction=False,
        independent_confirmation=False, risk_calibrated=False, closed_role_readout=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(reports/'analysis.json', result)
    if verify:
        immutable_json(reports/'replay.json', dict(analysis_sha256=file_digest(reports/'analysis.json'),
            checkpoints_replayed=len(views), prediction_rows=result['prediction_rows'],
            matched_samplers=len(views), all_checks_passed=True))
    beat(state='verified' if verify else 'evaluated', analysis_sha256=file_digest(reports/'analysis.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'resume', 'evaluate', 'verify'):
        parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--trial')
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if sum((args.audit_only, args.evaluate, args.verify)) > 1 or (
            args.stop_at is not None and (not args.trial or args.evaluate or args.verify)):
        raise ValueError('One explicit phase; pilot names one registered fit')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, rows, views, identity = load()
    if args.trial and args.trial not in views:
        raise ValueError('Unregistered fit')
    root = ROOT/cfg['output']
    root.mkdir(parents=True, exist_ok=True)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), fits=len(views))
        elif args.evaluate or args.verify:
            evaluate(cfg, data, rows, views, identity, beat, verify=args.verify)
        else:
            for key, view in views.items():
                if args.trial and args.trial != key:
                    continue
                folder, ti = root/'trials'/key, trial_identity(identity, key, view)
                if (folder/'complete.json').exists():
                    completed(folder/'complete.json', ti, cfg['training'])
                    beat(state='cached_verified_complete', trial=key)
                    continue
                torch.manual_seed(view['seed'])
                model = build_forecaster(cfg['architecture'])
                beat(state='training_started', trial=key, training_rows=len(view['fold']['train_ids']))
                fit = fit_trial(model, data, view['fold'], seed=view['seed'], settings=cfg['training'],
                    identity=ti, directory=folder, resume=args.resume, stop_at=args.stop_at,
                    heartbeat=lambda **v:beat(trial=key, **v))
                if fit['complete']:
                    assert_current(identity)
                    immutable_json(folder/'complete.json', dict(identity=ti, fit=fit,
                        checkpoint=str((folder/'checkpoint.pt').relative_to(ROOT)),
                        checkpoint_sha256=file_digest(folder/'checkpoint.pt')))
                else:
                    beat(state='pilot_completed_not_full', trial=key)
                    return
            beat(state='training_call_complete')


if __name__ == '__main__':
    main()
