"""Diagnose sealed subset-risk actions without training or selecting a policy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before importing Torch')
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_subset_excess as parent
from src.world_model import m3w_selected_risk_diagnosis as api
base = parent.base
PUBLIC = parent.PUBLIC.parent/'european_selected_risk_diagnosis_v1'
PRIVATE = parent.PRIVATE.parent/'european_selected_risk_diagnosis_v1'
CONFIG = 'configs/m3w_european_selected_risk_diagnosis_v1.json'
FILES = [CONFIG, 'scripts/diagnose_m3w_selected_risk.py',
         'src/world_model/m3w_selected_risk_diagnosis.py',
         'tests/test_m3w_selected_risk_diagnosis.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def identity():
    cfg = json.loads((ROOT/CONFIG).read_text())
    path = parent.PUBLIC/'verification.json'
    assert base.digest(path) == cfg['parent_seal_sha256']
    seal = json.loads(path.read_text())
    for p, h in seal['source_bindings'].items():
        assert base.digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items():
        assert base.digest(parent.PUBLIC/p) == h, p
    assert not any(cfg[k] for k in ('new_training', 'new_policy', 'threshold_search',
        'independent_roles_read', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    assert cfg['support_quantile'] == .95 and cfg['groups'] == 108
    return cfg, dict(parent_seal=base.artifact(path), bindings={p:base.digest(ROOT/p) for p in FILES},
                     posthoc_development_diagnostic=True, independent_roles_read=False)


def diagnose_group(c, data, causal, pair, ref, parent_identity):
    assert base.artifact(ROOT/ref['path']) == ref
    doc = json.loads((ROOT/ref['path']).read_text())
    assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
    with np.load(ROOT/doc['arrays']['path'], allow_pickle=False) as z:
        actions = {k:z[k].copy() for k in z.files}
    held, scores, pr = parent.predictions(c, causal, doc['fit'], parent_identity)
    ids = c['ids']; np.testing.assert_array_equal(ids[held], actions['ids'])
    fitdoc = json.loads((ROOT/doc['fit']['path']).read_text())
    roles = fitdoc['identity']['roles_and_targets']
    base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
    fit = np.flatnonzero(np.isin(data['sites'][ids], roles['training_sites']))
    u = parent.api.head.descriptors(data['geometry'][ids], c['floor'], c['prediction'], c['x'], pr)
    bank = parent.masks(c, causal)
    size = api.query_sizes(data['sites'][ids], data['recordings'][ids], data['frames'][ids])
    state = parent.api.head.read_checkpoint(ROOT/fitdoc['artifacts']['subset_pointwise']['path'])
    np.testing.assert_array_equal(state['subsets'], bank[fit])
    sup = api.support_from_fit(u[fit], data['sites'][ids[fit]], pr['known'], u[held],
                               state['descriptor_preprocess']['mean'], state['descriptor_preprocess']['std'])
    # Held outcomes are first used after all causal scores, support and masks exist.
    cv, _, (floor, _), (neural, _) = base.floor_api.costs(c, data, np.arange(len(ids)))
    truth = api.signed_cost(cv, floor, neural, c['job']['design']['easy_cut'], pr['cost_scale'])
    np.testing.assert_array_equal(pr['known'], np.isfinite(truth[fit]).all(1))
    basis = base.floor_api.api.targets(cv, floor, neural, c['job']['design']['easy_cut'])[:, [2,1,3,4]]
    np.testing.assert_allclose(truth, parent.api.head.parent.signed(basis)/pr['cost_scale'], rtol=1e-12, atol=1e-12)
    rows, queries, exchanges, fitting = [], [], [], []
    common = dict(group=c['name']+f'_pair{pair}', seed=c['job']['old_identity']['seed'])
    for arm, prediction in scores.items():
        q = parent.api.head.parent.signed(prediction.astype(float))/pr['cost_scale']
        s = parent.api.head.read_checkpoint(ROOT/fitdoc['artifacts'][arm]['path'])
        model = parent.previous.descriptor.model_from(s)
        train_p = parent.api.head.predict(model, c['x'][fit], u[fit], c['env'][fit], pr, s['descriptor_preprocess'])
        train_q = parent.api.head.parent.signed(train_p.astype(float))/pr['cost_scale']
        for site in roles['training_sites']:
            at = data['sites'][ids[fit]] == site
            m = api.residual_metrics(train_q[at], truth[fit][at], np.ones(at.sum(), bool), bank[fit][at],
                                     np.zeros(at.sum(), bool), size[fit][at])
            fitting.append(dict(**common, arm=arm, site=site, metric=m, in_sample=True))
        for site in roles['held_sites']:
            at = data['sites'][ids[held]] == site
            ix = held[at]; eligible = actions['eligible'][at]; b = bank[ix]
            outside = sup['outside_both'][at]
            masks = {'eligible':eligible, **{name:b[:, j] for j, name in enumerate(parent.api.SUBSETS)}}
            for policy in ('joint', 'rank'):
                take = actions[arm+'_'+policy][at]
                assert not (take & ~eligible).any()
                masks.update({policy+'_selected':take, policy+'_unselected_eligible':eligible & ~take,
                    policy+'_selected_in_proxy':take & b[:, 0], policy+'_selected_out_proxy':take & ~b[:, 0],
                    policy+'_selected_in_support':take & ~outside, policy+'_selected_out_support':take & outside})
                qm = api.query_residuals(q[at], truth[ix], take, data['recordings'][ids[ix]], data['frames'][ids[ix]])
                queries.append(dict(**common, site=site, arm=arm, policy=policy, metric=qm))
            for key, mask in masks.items():
                m = api.residual_metrics(q[at], truth[ix], mask, b, outside, size[ix])
                m['outside_either_fit_sources_fraction'] = float(sup['outside_either'][at][mask].mean()) if mask.any() else None
                rows.append(dict(**common, site=site, arm=arm, slice=key, metric=m))
    for site in roles['held_sites']:
        at = data['sites'][ids[held]] == site; ix = held[at]
        for policy in ('joint', 'rank'):
            before, after = (actions[a+'_'+policy][at] for a in parent.api.ARMS)
            if policy == 'rank':
                assert before.sum() == after.sum()
            m = api.exchange_accounting(floor[ix], neural[ix], before, after)
            exchanges.append(dict(**common, site=site, policy=policy, metric=m))
    return dict(parent_action=ref, parent_fit=doc['fit'], rows=rows, queries=queries, exchanges=exchanges,
        fitting=fitting, support=dict(thresholds=sup['thresholds'], fitting_sites=sup['fitting_sites'],
            descriptors_hash=base.inter.array_hash(u), support_hash=base.inter.array_hash(sup['outside_both']),
            fitting_known_rows=int(pr['known'].sum()), fit_ids_hash=base.inter.array_hash(ids[fit])),
        future_fields_removed_from_inference=True, new_actions=False)


def locality_reduce(rows, metric, localities, bootstrap=False):
    values = {}; missing = 0
    for site in localities:
        block = [r['metric'][metric] for r in rows if r['site'] == site]
        valid = [x for x in block if x is not None and np.isfinite(x)]
        missing += len(block)-len(valid)
        values[site] = float(np.mean(valid)) if valid else None
    valid = [v for v in values.values() if v is not None]
    out = dict(mean=float(np.mean(valid)) if valid else None, by_locality=values,
        observed_localities=len(valid), expected_localities=len(localities), undefined_views=missing, ci95=None)
    if bootstrap and not missing and len(valid) == len(localities):
        rng = np.random.default_rng(101531)
        means = np.asarray(valid)[rng.integers(0, len(valid), size=(3000, len(valid)))].mean(1)
        out['ci95'] = np.quantile(means, [.025, .975]).tolist()
    return out


def summarize(records, ident):
    rows = [r for d in records for r in d['rows']]
    queries = [r for d in records for r in d['queries']]
    exchanges = [r for d in records for r in d['exchanges']]
    fitting = [r for d in records for r in d['fitting']]
    localities = sorted({r['site'] for r in rows}); assert len(localities) == 12
    def reduce_keyed(data, keys, bootstrap=False):
        grouped = {}
        for r in data:
            grouped.setdefault('/'.join(str(r[k]) for k in keys), []).append(r)
        return {k:{m:locality_reduce(rr, m, localities, bootstrap) for m in rr[0]['metric']} for k, rr in grouped.items()}
    lookup = {(r['group'], r['site'], r['arm'], r['slice']):r for r in rows}
    contrasts = []
    for r in rows:
        if r['slice'] not in ('joint_selected', 'rank_selected'):
            continue
        policy = r['slice'].split('_')[0]
        other = lookup[(r['group'], r['site'], r['arm'], policy+'_unselected_eligible')]
        metrics = {}
        for axis in ('all', 'easy'):
            key = axis+'_optimism_mean'; a, b = r['metric'][key], other['metric'][key]
            metrics[key+'_selected_minus_unselected'] = None if a is None or b is None else a-b
        contrasts.append(dict(group=r['group'], site=r['site'], arm=r['arm'], policy=policy, metric=metrics))
    return dict(identity=ident, result_source='fresh_run_frozen_action_diagnosis',
        parent_models_and_actions='cached_verified', groups=len(records), localities=localities,
        residuals=reduce_keyed(rows, ('arm', 'slice')), query_residuals=reduce_keyed(queries, ('arm', 'policy')),
        fitting_in_sample=reduce_keyed(fitting, ('arm',)),
        selected_residual_contrasts=reduce_keyed(contrasts, ('arm', 'policy'), True),
        exchanges=reduce_keyed(exchanges, ('policy',), True),
        totals=dict(held_residual_views=len(rows), query_views=len(queries), exchange_views=len(exchanges),
            fitting_views=len(fitting), predicted_safe_actual_excess_queries={a:{p:{axis:sum(r['metric'][axis+'_predicted_safe_actual_excess_queries']
                for r in queries if r['arm']==a and r['policy']==p) for axis in ('all','easy')} for p in ('joint','rank')} for a in parent.api.ARMS}),
        independent_confirmation=False, new_training=False, new_policy=False, uncertainty_calibrated=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    cfg, ident = identity()
    if args.register:
        base.immutable_json(PUBLIC/'registration.json', ident); print(json.dumps(ident)); return
    assert json.loads((PUBLIC/'registration.json').read_text()) == ident
    base.inter.committed(PUBLIC/'registration.json')
    base.torch.set_num_threads(4); base.torch.set_num_interop_threads(1)
    started = time.monotonic()
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        beat('started', replay=args.replay)
        _, data, jobs, oid, _, _, _, _, _, pid = parent.load()
        causal = {k:data[k] for k in parent.CAUSAL_KEYS}
        freeze = json.loads((parent.PUBLIC/'decision_freeze.json').read_text())
        refs = {Path(r['path']).stem:r for r in freeze['groups']}; records = []; artifacts = []
        for c in base.floor_api.contexts(causal, jobs, oid):
            for pair in range(6):
                name = c['name']+f'_pair{pair}'; path = PRIVATE/'groups'/(name+'.json')
                if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes']:
                    raise OSError('Keep10GiB reserve and completed diagnostic groups')
                if path.exists() and args.resume and not args.replay:
                    d = json.loads(path.read_text()); assert d['identity'] == ident
                    assert base.artifact(ROOT/d['parent_action']['path']) == d['parent_action']
                else:
                    if path.exists() and not args.replay:
                        raise ValueError('Existing groups require --resume or --replay')
                    d = dict(identity=ident, **diagnose_group(c, data, causal, pair, refs[name], pid))
                    base.immutable_json(path, d)
                records.append(d); artifacts.append(base.artifact(path))
                beat('group_replayed' if args.replay else 'group_complete', group=name, complete=len(records))
        assert len(records) == cfg['groups']
        result = summarize(records, ident)
        base.immutable_json(PUBLIC/'summary.json', result)
        base.immutable_json(PUBLIC/('replay.json' if args.replay else 'completion.json'),
            dict(identity=ident, groups=artifacts, summary=base.artifact(PUBLIC/'summary.json'),
                 exact_replay=args.replay, seconds=time.monotonic()-started, pid=os.getpid(),
                 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 independent_roles_read=False, new_training=False, new_policy=False))
        beat('complete', groups=len(records), seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
