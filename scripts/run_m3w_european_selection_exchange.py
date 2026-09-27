"""Decompose frozen same-query selection differences without fitting a policy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_causal_descriptor_refit as parent
from src.world_model import m3w_selection_exchange as api

PUBLIC = parent.PUBLIC.parent/'european_selection_exchange_v1'
PRIVATE = parent.PRIVATE.parent/'european_selection_exchange_v1'
CONFIG = 'configs/m3w_european_selection_exchange_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_selection_exchange.py',
    'scripts/run_m3w_european_selection_exchange.py', 'tests/test_m3w_selection_exchange.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state, **values):
    entry = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    parent.base.inter.json_write(PRIVATE/'heartbeat.json', entry)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(entry)+'\n')
    print(json.dumps(entry), flush=True)


def load(register=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    sealpath = parent.PUBLIC/'verification.json'
    assert parent.base.digest(sealpath) == cfg['parent_seal_sha256']
    seal = json.loads(sealpath.read_text())
    for path, h in seal['source_bindings'].items():
        assert parent.base.digest(ROOT/path) == h, path
    for path, h in seal['artifacts'].items():
        assert parent.base.digest(parent.PUBLIC/path) == h, path
    _, data, jobs, oid, _, _, _, _ = parent.load()
    assert cfg['groups'] == 108 and cfg['policies'] == ['descriptor', 'control_matched_count']
    assert not any(cfg[k] for k in ('new_training', 'decision_changes', 'held_threshold_search',
        'independent_roles_read', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent_seal=parent.base.artifact(sealpath),
        bindings={p: parent.base.digest(ROOT/p) for p in FILES},
        localities=sorted(set(data['sites'])), source_rows=len(data['sites']),
        new_training=False, decisions_frozen=True, independent_roles_read=False)
    if register:
        parent.base.immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        parent.base.inter.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, oid, identity


def run(cfg, data, jobs, oid, identity, *, resume=False, replay=False, limit=None):
    freeze = json.loads((parent.PUBLIC/'decision_freeze.json').read_text())
    refs = {Path(r['path']).parent.name: r for r in freeze['heads']}
    previous = json.loads((parent.PRIVATE/'details.json').read_text())
    parent_metrics = {(r['group'], r['site']): r['metric'] for r in previous['rows']
        if r['policy'] == 'control_matched_count'}
    records, outputs = [], []
    for c in parent.base.floor_api.contexts(data, jobs, oid):
        _, _, (floor, _), (neural, _) = parent.base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        assert np.all(np.diff(c['ids']) > 0)
        for pair in range(6):
            name = c['name']+f'_pair{pair}'
            home = PRIVATE/'groups'/(name+'.json')
            ref = refs[name]
            assert parent.base.artifact(ROOT/ref['path']) == ref
            rec = json.loads((ROOT/ref['path']).read_text())
            if home.exists() and resume and not replay:
                doc = json.loads(home.read_text())
                assert doc['identity'] == identity and doc['parent_head'] == ref
            else:
                if home.exists() and not replay:
                    raise ValueError('Use --resume to preserve completed groups')
                if shutil.disk_usage(PRIVATE).free < 10*2**30:
                    raise OSError('Preserve10GiB and completed results')
                for a in rec['artifacts'].values():
                    assert parent.base.artifact(ROOT/a['path']) == a
                with np.load(ROOT/rec['artifacts']['decisions']['path'], allow_pickle=False) as z:
                    a = {k: z[k].copy() for k in z.files}
                state = parent.api.read_checkpoint(ROOT/rec['artifacts']['checkpoint']['path'])
                scale = state['preprocess']['cost_scale']
                roles = rec['identity']['roles_and_targets']
                parent.base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'],
                    roles['training_sites'], roles['held_sites'])
                ridgepath = parent.base.floor_api.PRIVATE/'fits'/name/'complete.json'
                ridge = json.loads(ridgepath.read_text())
                for r in ridge['artifacts'].values():
                    assert parent.base.artifact(ROOT/r['path']) == r
                with np.load(ROOT/ridge['artifacts']['scores']['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], a['ids'])
                    utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))
                ids = a['ids']; pos = np.searchsorted(c['ids'], ids)
                np.testing.assert_array_equal(c['ids'][pos], ids)
                keys = np.array([str(s)+'|'+str(r) for s, r in zip(data['sites'][ids], data['recordings'][ids])])
                query = api.check_queries(a['descriptor'], a['control_matched_count'], a['eligible'],
                    keys, data['frames'][ids], ids)
                rows = []
                for site in roles['held_sites']:
                    at = data['sites'][ids] == site
                    value = api.account(floor[pos][at], neural[pos][at], a['descriptor'][at],
                        a['control_matched_count'][at], a['eligible'][at], utility[at], scale)
                    old = parent_metrics[(name, site)]
                    np.testing.assert_allclose(value['old_error_sum']*scale, old['error_sum'], rtol=1e-12)
                    rows.append(dict(group=name, site=site, seed=roles['seed'] if 'seed' in roles else c['job']['old_identity']['seed'],
                        scale=scale, **value))
                doc = dict(identity=identity, parent_head=ref, utility_source=parent.base.artifact(ridgepath),
                    queries=query, rows=rows)
                parent.base.immutable_json(home, doc)
            outputs.append(parent.base.artifact(home)); records.extend(doc['rows'])
            beat('group_replayed' if replay else 'group_complete', group=name, complete=len(outputs))
            if limit and len(outputs) >= limit:
                return
    assert len(outputs) == cfg['groups']
    reduced = {k: parent.base.inter.paired_localities(records, identity['localities'], k,
        cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in records[0]['metric']}
    old_summary = json.loads((parent.PUBLIC/'summary.json').read_text())
    contrast = old_summary['paired']['control_matched_count']['ADE_gain_percent']
    for key in ('point', 'ci95'):
        np.testing.assert_allclose(reduced['net_ADE_gain_percent'][key], contrast[key], atol=1e-11)
    summary = dict(identity=identity, result_source='fresh_frozen_policy_exchange_accounting',
        groups=len(outputs), held_views=len(records), metrics=reduced,
        parent_contrast_reconstructed=True, primary_replaced=False, new_training=False,
        decision_changes=False, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    parent.base.immutable_json(PUBLIC/'summary.json', summary)
    parent.base.immutable_json(PUBLIC/'completion.json', dict(identity=identity, groups=outputs,
        summary=parent.base.artifact(PUBLIC/'summary.json')))
    if replay:
        parent.base.immutable_json(PUBLIC/'replay.json', dict(groups=len(outputs), exact=True,
            parent_contrast_reconstructed=True, new_training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['register', 'run', 'replay'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('Positive group limit required')
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.base.torch.set_num_threads(4); parent.base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('started', phase=args.phase)
        values = load(args.phase == 'register')
        if args.phase != 'register':
            run(*values, resume=args.resume, replay=args.phase == 'replay', limit=args.limit)
        beat('complete', phase=args.phase)


if __name__ == '__main__':
    main()
