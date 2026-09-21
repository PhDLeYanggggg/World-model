"""Registered continuous-cost training; frozen decisions precede label readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_joint_controls import load as parent_load, view_scores
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write, array_hash
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_forecast_cost_bounds import disagreement, bounded_fractions, partial_gain_bounds
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.world_model.m3w_native_gain_harm import cost_features, preprocess
from src.world_model.m3w_bounded_cost_head import ARMS, build, fit, predict
import numpy as np
import torch

CONFIG = 'configs/m3w_bounded_cost_v1.json'
CODE = ('scripts/run_m3w_bounded_cost.py', 'src/world_model/m3w_bounded_cost_head.py',
        'src/evaluation/m3w_forecast_cost_bounds.py', 'tests/test_m3w_bounded_cost_head.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    _, data, _, views, gain, _, parent = parent_load()
    if (tuple(cfg['arms']) != ARMS or cfg['sites'] != ['coupa', 'deathCircle', 'gates', 'hyang']
            or cfg['seeds'] != [17, 29, 43] or cfg['training']['steps'] != 3000
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'independent_confirmation',
                'risk_calibration', 'closed_role_readout', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only design required')
    bindings = dict(parent['source_bindings'])
    path = 'outputs/publication_readiness_2026_09/native_cost_bounds_v1/analysis.json'
    assert file_digest(ROOT/path) == cfg['diagnostic_sha256']
    for p in (path, CONFIG, cfg['registration'], *CODE):
        bindings[p] = file_digest(ROOT/p)
    for meta in views.values():
        for kind in ('inputs', 'targets'):
            path = meta[kind+'_path']
            assert file_digest(ROOT/path) == meta[kind+'_sha256']
            bindings[path] = meta[kind+'_sha256']
    assert_current(dict(source_bindings=bindings))
    identity = dict(source_bindings=bindings, config=cfg, torch=torch.__version__, numpy=np.__version__,
        architecture=platform.machine(), runtime=dict(torch_threads=4, interop_threads=1, num_workers=0))
    return cfg, data, views, gain, identity


def features(geometry, candidate, scale):
    x, same = cost_features(geometry, candidate, scale)
    d = disagreement(candidate, geometry[:, 332:356].reshape(-1, 12, 2), scale).mean(1)
    return np.column_stack((x, np.log1p(d))).astype(np.float32), d, same


def read_arrays(data, ids, field):
    shape = (len(ids), 12) if field == 'valid' else (len(ids), 12, 2)
    out = np.empty(shape, bool if field == 'valid' else np.float32)
    for rec in np.unique(data['recordings'][ids]):
        all_ids = np.flatnonzero(data['recordings'] == rec)
        positions = np.flatnonzero(data['recordings'][ids] == rec)
        local = np.searchsorted(all_ids, ids[positions])
        np.testing.assert_array_equal(all_ids[local], ids[positions])
        path = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'/rec/(field+'.npy')
        out[positions] = np.load(path, mmap_mode='r', allow_pickle=False)[local]
    return out


def training_data(meta, data):
    with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
        ids, p = z['ids'].copy(), z['prediction'].copy()
    with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        y, cv = np.column_stack((z['benefit'], z['harm'])), z['baseline_ade'].copy()
    np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] != meta['outer_site']))
    x, d, _ = features(data['geometry'][ids], p, data['scale'][ids])
    full = read_arrays(data, ids, 'valid').all(1)
    bounded_fractions(y, d, full)
    y[~full], cv[~full] = np.nan, np.nan
    pr = preprocess(x, y, cv, data['sites'][ids], meta['outer_site'])
    assert x.shape[1] == 356 and np.array_equal(full, pr['known'])
    return ids, x, y, d, pr


def train(cfg, data, views, identity, args, beat):
    for key, meta in views.items():
        if args.view and key != args.view:
            continue
        ids, x, y, d, pr = training_data(meta, data)
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d),
            labels_sha256=array_hash(y), complete_mask_sha256=array_hash(pr['known']),
            training_sites=pr['training_sites'], producers=meta['training_producers'])
        for arm in ARMS:
            if args.arm and arm != args.arm:
                continue
            folder = ROOT/cfg['output']/'trials'/key/arm
            receipt = folder/'complete.json'
            if receipt.exists():
                r = json.loads(receipt.read_text())
                assert r['identity'] == ti and r['fit']['complete'] and file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_complete', view=key, arm=arm)
                continue
            _, result = fit(x, y, d, data['sites'][ids], pr, arm=arm, seed=meta['seed'],
                settings=cfg['training'], identity=ti, directory=folder, resume=args.resume,
                stop_at=args.stop_at, heartbeat=lambda **v:beat(view=key, arm=arm, **v))
            if not result['complete']:
                beat(state='pilot_complete_not_full_fit', view=key, arm=arm)
                return
            cp = folder/'checkpoint.pt'
            assert_current(identity)
            immutable_json(receipt, dict(identity=ti, arm=arm, fit=result, rows=len(ids), supported_rows=int(pr['known'].sum()),
                checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp), independent_confirmation=False))
    beat(state='training_call_complete')


def selections(score, past, distance, legacy, ids):
    support = ~np.all(past[:, -1] == past[:, -2], axis=1) & (distance > 0)
    gain = score[:, 0]-score[:, 1]
    net = support & (gain > 0)
    strict = net & (score[:, 1] <= .1*score[:, 0])
    count = int(legacy.sum())
    pool = np.flatnonzero(support)
    if len(pool) < count:
        raise ValueError('Past-only matched capacity insufficient')
    selected = pool[np.lexsort((ids[pool], -gain[pool]))[:count]]
    matched = np.zeros(len(ids), bool)
    matched[selected] = True
    return dict(net_stop=net, strict_stop=strict, matched_count=matched)


def evaluate(cfg, data, views, gain, identity, beat, verify):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    records = {}
    for key in views:
        for arm in ARMS:
            r = json.loads((root/'trials'/key/arm/'complete.json').read_text())
            assert r['identity']['identity'] == identity and r['fit']['complete'] and r['fit']['step'] == cfg['training']['steps']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            records[key, arm] = r
    n = len(data['sites'])
    names = ['floor', 'uncontrolled', 'legacy_stop_mse_strict']+[a+'_'+p for a in ARMS for p in cfg['policies']]
    chosen = {s:{name:np.zeros(n, bool) for name in names} for s in cfg['seeds']}
    predictions, all_scores, prs, archives = {}, {}, {}, []
    replays = 0
    for key, meta in views.items():
        ids, p, b, _, legacy, parent_pr = view_scores(key, meta, gain, data)
        x, d, same = features(data['geometry'][ids], p, data['scale'][ids])
        seed = meta['seed']
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        arrays = dict(ids=ids)
        predictions[key], prs[key] = (ids, p, b, d), parent_pr
        chosen[seed]['uncontrolled'][ids] = True
        chosen[seed]['legacy_stop_mse_strict'][ids] = legacy
        draws = None
        for arm in ARMS:
            cp = torch.load(ROOT/records[key, arm]['checkpoint'], map_location='cpu', weights_only=False)
            assert cp['identity'] == records[key, arm]['identity'] and cp['step'] == cfg['training']['steps'] and cp['arm'] == arm
            assert cp['seed'] == seed and cp['settings'] == cfg['training']
            assert cp['draws'][~cp['preprocess']['known']].sum() == 0 and cp['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
            if draws is None:
                draws = cp['draws']
            else:
                np.testing.assert_array_equal(draws, cp['draws'])
            model = build(x.shape[1], cfg['training']['width'], seed)
            model.load_state_dict(cp['model'])
            score = predict(model, x, d, cp['preprocess'], arm)
            assert np.isfinite(score).all() and np.all(score >= 0) and not score[same].any()
            if arm != 'direct_native':
                assert np.all(score.sum(1) <= d+2e-6*(1+d))
            arrays[arm] = score
            all_scores[key, arm] = score
            for policy, bits in selections(score, past, d, legacy, ids).items():
                arrays[arm+'_'+policy] = bits
                chosen[seed][arm+'_'+policy][ids] = bits
            replays += len(ids)
        path = root/'scores'/f'{key}.npz'
        if verify and not path.exists():
            raise ValueError('Cannot verify missing decision archive')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
    # Every policy and model endpoint is fixed before loading evaluation outcomes.
    selection_receipt = dict(identity=identity, archives=archives, score_rows=replays, future_target_used_in_decisions=False)
    immutable_json(root/'decisions_complete.json', selection_receipt)
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale'])
    full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    ne, nf = {s:np.full(n, np.nan) for s in cfg['seeds']}, {s:np.full(n, np.nan) for s in cfg['seeds']}
    lower, upper = {s:np.zeros(n) for s in cfg['seeds']}, {s:np.zeros(n) for s in cfg['seeds']}
    fit_quality = []
    for key, meta in views.items():
        ids, p, b, d = predictions[key]
        seed = meta['seed']
        ne[seed][ids], nf[seed][ids] = native_errors(p, y[ids], valid[ids], data['scale'][ids])
        bounds = partial_gain_bounds(p, b, y[ids], valid[ids], data['scale'][ids])
        lower[seed][ids], upper[seed][ids] = bounds['lower'], bounds['upper']
        masks['hard'][ids] = cv[ids] >= prs[key]['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= prs[key]['positive_easy_cut'])
        delta = cv[ids]-ne[seed][ids]
        costs = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
        for arm in ARMS:
            score = all_scores[key, arm]
            use = full[ids]
            strict = chosen[seed][arm+'_strict_stop'][ids] & use
            fit_quality.append(dict(view=key, arm=arm, complete_native_cost_MSE=float(((score[use]-costs[use])**2).mean()),
                strict_complete_rows=int(strict.sum()),
                strict_predicted_harm=None if not strict.any() else float(score[strict, 1].mean()),
                strict_observed_harm=None if not strict.any() else float(costs[strict, 1].mean())))
    def metric(model, reference, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(model[mask], reference[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in names:
        seeds, errors, endpoints = {}, [], []
        for seed in cfg['seeds']:
            use = chosen[seed][name]
            ade, fde = np.where(use, ne[seed], cv), np.where(use, nf[seed], cf)
            errors.append(ade)
            endpoints.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={site:[float(np.where(use, bound, 0)[data['sites'] == site].mean()) for bound in (lower[seed], upper[seed])] for site in cfg['sites']},
                subsets={k:metric(ade, cv, mask) for k,mask in masks.items()})
        mean = np.mean(errors, axis=0)
        summaries[name] = dict(ADE=metric(mean, cv), FDE=metric(np.mean(endpoints, axis=0), cf), seeds=seeds,
            subsets={k:metric(mean, cv, mask) for k,mask in masks.items()})
    contrasts = {}
    for policy in cfg['policies']:
        for left, right in (('bounded_native', 'direct_native'), ('bounded_fraction', 'bounded_native')):
            a, b = summaries[left+'_'+policy]['ADE']['by_scene'], summaries[right+'_'+policy]['ADE']['by_scene']
            contrasts[left+'_minus_'+right+'_'+policy] = paired_scene_contrast([a[s]['gain_percent'] for s in cfg['sites']], [b[s]['gain_percent'] for s in cfg['sites']])
    result = dict(identity=identity, result_source='fresh_run_36_continuous_cost_heads_and_fixed_source_readout',
        training=[dict(view=key, arm=arm, **{k:r[k] for k in ('fit', 'rows', 'supported_rows', 'checkpoint', 'checkpoint_sha256')}) for (key,arm),r in records.items()],
        archives=archives, summaries=summaries, contrasts=contrasts, conditional_quality=fit_quality,
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'), score_rows=replays,
        independent_confirmation=False, risk_calibrated=False, closed_role_readout=False, deployment=False,
        matched_count_is_offline_diagnostic=True, primary_metric_changed=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'verification_with_replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            checkpoints_replayed=len(records), score_rows_replayed=replays, matched_sampler_views=len(views),
            all_checks_passed=True, independent_confirmation=False))
    beat(state='verified' if verify else 'evaluated', analysis_sha256=file_digest(public/'analysis.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'evaluate', 'verify', 'resume'):
        parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--view')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if sum((args.audit_only, args.evaluate, args.verify)) > 1:
        raise ValueError('One explicit phase required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, data, views, gain, identity = load()
    if args.view and args.view not in views:
        raise ValueError('Unregistered view')
    folder = ROOT/cfg['output']
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder/'identity.json', identity)
    def beat(**values):
        row = dict(pid=os.getpid(), updated_unix=time.time(), **values)
        json_write(folder/'heartbeat.json', row)
        print(json.dumps(row), flush=True)
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']))
        elif args.evaluate or args.verify:
            evaluate(cfg, data, views, gain, identity, beat, args.verify)
        else:
            train(cfg, data, views, identity, args, beat)


if __name__ == '__main__':
    main()
