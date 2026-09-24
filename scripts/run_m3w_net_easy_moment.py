"""Frozen predictor, fresh signed easy-moment fitting and outcome-blind allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch

from scripts import run_m3w_easy_moment as parent
from scripts.run_m3w_easy_allocation_guarded import load as load_allocation
from scripts.run_m3w_easy_allocation import causal_view, ARMS as OLD_ARMS
from scripts.run_m3w_bounded_cost import training_data, features, read_arrays
from scripts.run_m3w_native_forecast import array_hash, file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_protected_motion_controls import causal_candidate, supported_costs
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_native_scene_alignment import scene_index
from src.training.m3w_easy_moment import fit, predict
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_net_easy_risk import targets, moments, allocate

CONFIG = 'configs/m3w_net_easy_moment_v1.json'
CODE = ('scripts/run_m3w_net_easy_moment.py', 'src/world_model/m3w_net_easy_risk.py',
        'tests/test_m3w_net_easy_risk.py', 'src/training/m3w_easy_moment.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    allocation = load_allocation()
    training = parent.load()
    frozen = ROOT/'data/stage_cvpr2027_experiments/easy_moment_v1/identity.json'
    if training[6] != json.loads(frozen.read_text()):
        raise ValueError('Training parent rebuilt identity differs from frozen source')
    old_path = ROOT/allocation[0]['reports']/'analysis.json'
    if file_digest(old_path) != cfg['parent_analysis_sha256']:
        raise ValueError('Changed allocation analysis')
    old = json.loads(old_path.read_text())
    diagnostic = 'outputs/publication_readiness_2026_09/easy_risk_definition_diagnosis_v1/analysis.json'
    assert file_digest(ROOT/diagnostic) == cfg['diagnosis_sha256']
    assert cfg['forest'] == training[0]['forest']
    for k in ('sites', 'seeds', 'actions'):
        assert cfg[k] == training[0][k] == allocation[0][k]
    assert cfg['easy_rho'] == .02 and cfg['bootstrap_resamples'] == 3000
    assert not any(cfg[k] for k in ('threshold_search', 'model_selection', 'new_forecast_training',
        'external_readout', 'independent_calibration', 'independent_confirmation',
        'deployment', 'stage5c_executed', 'smc_enabled'))
    bindings = dict(allocation[-1]['source_bindings'])
    for p, h in training[6]['source_bindings'].items():
        if p in bindings and bindings[p] != h:
            raise ValueError('Conflicting frozen source identity')
        bindings[p] = h
    for r in old['outcome_archives']:
        assert file_digest(ROOT/r['path']) == r['sha256']
        bindings[r['path']] = r['sha256']
    for p in (CONFIG, cfg['registration'], diagnostic, *CODE,
              str(old_path.relative_to(ROOT)), str(frozen.relative_to(ROOT)),
              'scripts/run_m3w_easy_allocation_guarded.py', 'src/evaluation/m3w_frozen_allocation_guard.py'):
        bindings[p] = file_digest(ROOT/p)
    identity = dict(config=cfg, source_bindings=bindings,
        parent_identity_sha256=file_digest(frozen),
        architecture=platform.machine(), runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        numpy=np.__version__, torch=torch.__version__, sklearn=__import__('sklearn').__version__,
        scipy=__import__('scipy').__version__, role='design_exposed_source_excluded_not_confirmation')
    assert_current(identity)
    return cfg, allocation, training, old, identity


def prepare(data, meta, action, cp, cutoff):
    # Preserve the frozen parent's support/preprocessing/draws, replace only its
    # label representation. Held-site outcomes are not read by this function.
    ids, x, y, d, pr = training_data(meta, data)
    with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); cv = z['baseline_ade'].copy()
    cv[~pr['known']] = np.nan
    if action == 'damped_velocity_005':
        g, s = data['geometry'][ids], data['scale'][ids]
        p = causal_candidate(g, action)
        x, d, _ = features(g, p, s)
        valid = read_arrays(data, ids, 'valid'); truth = read_arrays(data, ids, 'target')
        cv, _ = native_errors(g[:, 332:356].reshape(-1, 12, 2), truth, valid, s)
        err, _ = native_errors(p, truth, valid, s)
        y, cv = supported_costs(err, cv, valid.all(1))
        pr = preprocess(x, y, cv, data['sites'][ids], meta['outer_site'])
    for field in ('mean', 'std', 'known', 'weights', 'constant'):
        np.testing.assert_array_equal(pr[field], cp['preprocess'][field])
    assert pr['positive_easy_cut'] == cp['preprocess']['positive_easy_cut']
    assert cp['draws'].sum() == 768000 and not cp['draws'][~pr['known']].any()
    assert meta['outer_site'] not in pr['training_sites']
    assert not (data['sites'][ids] == meta['outer_site']).any()
    return ids, x, d, pr, targets(y, cv, d, pr['known'], cutoff)


def trial(cfg, key, action):
    return ROOT/cfg['output']/'trials'/key/action


def train(pack, args, beat):
    cfg, allocation, tp, old, identity = pack
    _, data, ev, mv, ep, cuts, _, frozen, ma, pa = tp
    ish = file_digest(ROOT/cfg['output']/'identity.json')
    for key in ev:
        if args.view and key != args.view:
            continue
        for action in cfg['actions']:
            if args.action and action != args.action:
                continue
            meta = ev[key] if action == 'eqmotion' else mv[key]
            cp = parent.old_head(frozen[key, action]); cutoff = cuts[key]['positive_easy_cut']
            ids, x, d, pr, q = prepare(data, meta, action, cp, cutoff)
            ti = dict(experiment_sha256=ish, view=key, action=action,
                inputs_sha256=array_hash(ids, x, d), targets_sha256=array_hash(q),
                draws_sha256=array_hash(cp['draws']), known_sha256=array_hash(pr['known']),
                training_sites=pr['training_sites'], easy_cut=cutoff,
                frozen_head_sha256=frozen[key, action]['checkpoint_sha256'])
            folder = trial(cfg, key, action); rp = folder/'complete.json'
            if rp.exists():
                r = json.loads(rp.read_text())
                assert r['identity'] == ti and r['fit']['complete']
                assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
                beat(state='cached_verified_fit', view=key, action=action)
                continue
            result = fit(x, q, d, pr, cp['draws'], settings=cfg['forest'], seed=meta['seed'],
                identity=ti, directory=folder, resume=args.resume, stop_at=args.stop_at,
                heartbeat=lambda **v: beat(view=key, action=action, **v))
            if not result['complete']:
                beat(state='pilot_checkpoint_not_full_matrix', view=key, action=action)
                return
            path = folder/'checkpoint.joblib'
            immutable_json(rp, dict(identity=ti, fit=result, checkpoint=str(path.relative_to(ROOT)),
                checkpoint_sha256=file_digest(path), result_source='fresh_run'))
    assert_current(identity)
    beat(state='all_requested_fits_complete')


def receipts(pack):
    cfg, ap, tp, old, identity = pack
    ish = file_digest(ROOT/cfg['output']/'identity.json'); result = {}
    for key in ap[4]:
        for action in cfg['actions']:
            r = json.loads((trial(cfg, key, action)/'complete.json').read_text())
            assert r['identity']['experiment_sha256'] == ish
            assert r['fit']['complete'] and r['fit']['trees'] == cfg['forest']['trees']
            assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
            assert key.rsplit('_seed', 1)[0] not in r['identity']['training_sites']
            result[key, action] = r
    return result


def decide(pack, args, beat):
    cfg, ap, tp, old, identity = pack
    data = ap[1]; root = ROOT/cfg['output']; ish = file_digest(root/'identity.json')
    fitted = receipts(pack); manifest = []; processed = 0
    for key in ap[4]:
        site, seedtext = key.rsplit('_seed', 1); seed = int(seedtext)
        for action in cfg['actions']:
            values, p, scale, cut = causal_view(ap, key, action); ids = values['ids']
            x, d, _ = features(data['geometry'][ids], p, data['scale'][ids])
            np.testing.assert_array_equal(d, values[action+'__distance'])
            cp = joblib.load(ROOT/fitted[key, action]['checkpoint'])
            assert cp['identity'] == fitted[key, action]['identity']
            f = predict(cp['model'], x, cp['preprocess']); qp, qn, r = moments(f, d, cut)
            g = values[action+'__score'][:, 0] - values[action+'__score'][:, 1]
            support = values[action+'__net_stop'] & (r > 0)
            index = scene_index(data['recordings'][ids], data['frames'][ids], data['tracks'][ids])
            oldref = next(v for v in old['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            # This archive contains causal choice bits only, not observed errors.
            with np.load(ROOT/oldref['path'], allow_pickle=False) as z:
                assert z.files == ['choices']; previous = z['choices'][ids].copy()
            for start in range(0, len(index['offsets'])-1, cfg['checkpoint_queries']):
                stop = min(len(index['offsets'])-1, start+cfg['checkpoint_queries'])
                path = root/'decisions'/key/action/f'{start:06d}.npz'; rp = path.with_suffix('.json')
                if rp.exists() and not args.verify:
                    record = json.loads(rp.read_text())
                    assert record['experiment_sha256'] == ish and file_digest(path) == record['sha256']
                    manifest.append(dict(path=str(rp.relative_to(ROOT)), sha256=file_digest(rp)))
                    processed += stop-start
                    continue
                if args.verify and not rp.exists():
                    raise ValueError('Replay cannot create missing decisions')
                local = index['order'][index['offsets'][start]:index['offsets'][stop]]
                choices = np.zeros((len(local), len(cfg['policies'])), bool)
                for j, name in enumerate(('net_stop', 'strict_stop', 'pointwise', 'aggregate_population')):
                    choices[:, j] = previous[local, OLD_ARMS.index(name)]
                choices[:, 4] = support[local] & (qp[local] <= cfg['easy_rho']*r[local])
                choices[:, 5] = support[local] & (qn[local] <= cfg['easy_rho']*r[local])
                reports = []; offset = 0
                for query in range(start, stop):
                    rows = index['order'][index['offsets'][query]:index['offsets'][query+1]]
                    budget = cfg['easy_rho']*float(r[rows].sum())
                    b, positive = allocate(g[rows], qp[rows], support[rows], budget, seconds=cfg['solver_seconds'])
                    n, signed = allocate(g[rows], qn[rows], support[rows], budget, seconds=cfg['solver_seconds'])
                    m, matched = allocate(g[rows], qn[rows], support[rows], budget,
                        count=int(b.sum()), seconds=cfg['solver_seconds'])
                    choices[offset:offset+len(rows), 6:] = np.column_stack((b, n, m)); offset += len(rows)
                    reports.append(dict(recording=str(index['recordings'][query]), frame=int(index['frames'][query]),
                        positive=positive, net=signed, matched=matched))
                write_arrays(path, dict(ids=ids[local], choices=choices, fractions=f[local], distance=d[local],
                    gain=g[local], support=support[local], positive_risk=qp[local], net_risk=qn[local], denominator=r[local]))
                record = dict(experiment_sha256=ish, view=key, action=action,
                    path=str(path.relative_to(ROOT)), sha256=file_digest(path), queries=reports)
                immutable_json(rp, record)
                manifest.append(dict(path=str(rp.relative_to(ROOT)), sha256=file_digest(rp)))
                processed += stop-start
                beat(state='past_only_decisions', view=key, action=action, queries=processed)
    assert processed == 188388
    assert_current(identity)
    immutable_json(root/'decisions_complete.json', dict(experiment_sha256=ish,
        query_action_seed_instances=processed, receipts=manifest,
        outcomes_used_for_decisions=False, all_fits_complete=True))
    if args.verify:
        immutable_json(ROOT/cfg['reports']/'decision_replay.json', dict(all_checks_passed=True,
            queries=processed, decision_manifest_sha256=file_digest(root/'decisions_complete.json')))
    beat(state='decisions_complete', queries=processed)


def evaluate(pack, args, beat):
    cfg, ap, tp, old, identity = pack
    data = ap[1]; parent_analysis = ap[3]; root = ROOT/cfg['output']; public = ROOT/cfg['reports']
    fitted = receipts(pack); n = len(data['sites']); arms = cfg['policies']
    completed = json.loads((root/'decisions_complete.json').read_text())
    assert completed['experiment_sha256'] == file_digest(root/'identity.json')
    bits = {(s, a): np.zeros((n, len(arms)), bool) for s in cfg['seeds'] for a in cfg['actions']}
    seen = {k: np.zeros(n, bool) for k in bits}; solver = []; quality = []
    for ref in completed['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); assert file_digest(ROOT/r['path']) == r['sha256']
        key = int(r['view'].rsplit('_seed', 1)[1]), r['action']
        with np.load(ROOT/r['path'], allow_pickle=False) as z:
            ids = z['ids']; assert not seen[key][ids].any()
            seen[key][ids] = True; bits[key][ids] = z['choices']
        solver.extend(dict(action=r['action'], view=r['view'], **q) for q in r['queries'])
    assert all(v.all() for v in seen.values())

    def metric(e, cv, mask=None, ci=False):
        mask = np.ones(n, bool) if mask is None else mask
        return paired_scene_metrics(e[mask], cv[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=3000 if ci else 0)

    summary = {}; contrasts = {}; outcomes = []
    for action in cfg['actions']:
        errors = {p: [] for p in arms}; ends = {p: [] for p in arms}; detail = {p: {} for p in arms}
        for seed in cfg['seeds']:
            ref = next(v for v in parent_analysis['outcome_archives'] if v['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/ref['path']) == ref['sha256']
            with np.load(ROOT/ref['path'], allow_pickle=False) as z:
                o = {k: z[k].copy() for k in z.files}
            cv, cf = o['cv'], o['cf']; masks = {k: o[k] for k in ('complete', 'zero_CV', 'positive_easy', 'hard')}
            for j, pol in enumerate(arms):
                selected = bits[seed, action][:, j]
                e = np.where(selected, o['candidate_ade'], cv); f = np.where(selected, o['candidate_fde'], cf)
                errors[pol].append(e); ends[pol].append(f)
                detail[pol][str(seed)] = dict(ADE=metric(e, cv), FDE=metric(f, cf),
                    subsets={k: metric(e, cv, m) for k, m in masks.items()}, selected=int(selected.sum()),
                    selected_unknown=int((selected & np.isnan(cv)).sum()),
                    selected_incomplete=int((selected & ~masks['complete']).sum()),
                    zero_CV_harmed=int((selected & masks['zero_CV'] & (e > 0)).sum()),
                    full_grid_gain_bounds={s: [float(np.where(selected, o[b], 0)[data['sites']==s].mean())
                        for b in ('lower', 'upper')] for s in cfg['sites']})
            op = root/'outcomes'/f'{action}_seed{seed}.npz'
            write_arrays(op, dict(choices=bits[seed, action])); outcomes.append(dict(path=str(op.relative_to(ROOT)), sha256=file_digest(op)))
        mean = {}
        for pol in arms:
            mean[pol] = np.mean(errors[pol], axis=0)
            summary[action+'__'+pol] = dict(ADE=metric(mean[pol], cv, ci=True),
                FDE=metric(np.mean(ends[pol], axis=0), cf, ci=True),
                subsets={k: metric(mean[pol], cv, m, True) for k, m in masks.items()}, seeds=detail[pol])
        contrasts[action] = {}
        for left, right in [('positive_point', 'old_positive_point'), ('positive_population', 'old_positive_population'),
                ('net_point', 'positive_point'), ('net_population', 'positive_population'),
                ('net_matched_positive', 'positive_population'), ('net_population', 'old_strict')]:
            result = {}
            for subset in ('all', 'hard', 'positive_easy'):
                mask = None if subset == 'all' else masks[subset]
                l, r = metric(mean[left], cv, mask)['by_scene'], metric(mean[right], cv, mask)['by_scene']
                result[subset] = paired_scene_contrast([l[s]['gain_percent'] for s in cfg['sites']],
                    [r[s]['gain_percent'] for s in cfg['sites']], resamples=3000)
            contrasts[action][left+'_minus_'+right] = result
    solver_summary = {a: {p: dict(queries=sum(q['action']==a for q in solver),
        failed=sum(q['action']==a and not q[p]['optimal'] for q in solver),
        risk_violations=sum(q['action']==a and not q[p]['constraint_pass'] for q in solver),
        count_failures=sum(q['action']==a and not q[p]['exact_count_pass'] for q in solver))
        for p in ('positive', 'net', 'matched')} for a in cfg['actions']}
    result = dict(result_source='fresh_moment_training_and_causal_decisions_cached_verified_forecasts_and_outcomes',
        experiment_sha256=file_digest(root/'identity.json'), decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        rows=n, sites=cfg['sites'], seeds=cfg['seeds'], policies=arms, summary=summary, contrasts=contrasts,
        solver=solver_summary, outcome_archives=outcomes,
        fits=[dict(view=k, action=a, **r) for (k, a), r in fitted.items()],
        independent_calibration=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if args.verify:
        immutable_json(public/'aggregate_replay.json', dict(all_checks_passed=True, analysis_sha256=file_digest(public/'analysis.json')))
    beat(state='evaluation_complete', rows=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'train', 'decide', 'evaluate'], default='preflight')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--view'); parser.add_argument('--action'); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    pack = load(); cfg = pack[0]; root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    if args.view and args.view not in pack[1][4]:
        raise ValueError('Unknown view')
    if args.action and args.action not in cfg['actions']:
        raise ValueError('Unknown action')
    immutable_json(root/'identity.json', pack[-1])

    def beat(**v):
        e = dict(pid=os.getpid(), updated_unix=time.time(), **v)
        json_write(root/'heartbeat.json', e)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(e)+'\n')
        print(json.dumps(e), flush=True)

    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'preflight':
            beat(state='preflight_pass', rows=len(pack[1][1]['sites']), bindings=len(pack[-1]['source_bindings']))
        elif args.phase == 'train':
            train(pack, args, beat)
        elif args.phase == 'decide':
            decide(pack, args, beat)
        else:
            evaluate(pack, args, beat)


if __name__ == '__main__':
    main()
