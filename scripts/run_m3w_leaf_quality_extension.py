"""Registered TRAIN-identity quality-extension control on fixed forecasts and heads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_cost_support_diagnostic as diagnostic
from scripts.m3w_bounded_packet_write import with_keepalive
from src.world_model import m3w_leaf_quality_extension as api

parent, prior, reader = diagnostic.parent, diagnostic.prior, diagnostic.reader
sha, once = diagnostic.sha, diagnostic.once
NAME = 'european_leaf_quality_extension_v1'
PUBLIC, PRIVATE = diagnostic.PUBLIC.parent/NAME, diagnostic.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')


class FrozenReader(reader.PositiveReader):
    def __init__(self, server):
        assert server in (diagnostic.SERVER, reader.SERVER, prior.previous.SERVER)
        self.server = server

    def __enter__(self):
        p = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(p/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((p/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.p = subprocess.Popen(with_keepalive(ssh)+[shlex.join(['/usr/bin/python3', '-c', self.server])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        try:
            assert self.line() == {'ready': True}
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self


def registration():
    assert diagnostic.registration() == json.loads((diagnostic.PUBLIC/'registration.json').read_text())
    receipt = json.loads((diagnostic.PUBLIC/'verification.json').read_text())
    for key in ('summary', 'complete'):
        assert sha(diagnostic.PUBLIC/(key+'.json')) == receipt[key+'_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_leaf_quality_extension'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_leaf_quality_extension.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        diagnostic_verification_sha256=sha(diagnostic.PUBLIC/'verification.json'),
        source_heads=72, new_training=False, known_training_predictions_unchanged=True,
        independent_roles_read=False, threshold_search=False)


def read_rows(path):
    result = {}
    for ref in json.loads((path/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        result[r['group'], r['head_seed']] = r
    assert len(result) == 72
    return result


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg)
        print('Registered leaf-quality extension with frozen TRAIN predictions')
        return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify':
        assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads'])
    parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cost_rows = read_rows(diagnostic.fitted_run.PUBLIC)
        poisson_rows = read_rows(prior.PUBLIC)
        docs, additive_rows = parent.parent.docs(), prior.prior_rows()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.prior.past_quality(data)
        assert provenance == json.loads((prior.prior.PUBLIC/'feature_provenance.json').read_text())
        rows, refs = [], []
        checks = fetched = size = 0
        with FrozenReader(diagnostic.SERVER) as cost_reader, FrozenReader(reader.SERVER) as poisson_reader, FrozenReader(prior.previous.SERVER) as add_reader:
            beat(state='owned_read_only_streams_connected', new_HPC_jobs=0)
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]
                    h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        beat(state='training_identity_and_frozen_extension', group=group, head_seed=seed)
                        source, anchor, po, ad = docs[group, seed], cost_rows[group, seed], poisson_rows[group, seed], additive_rows[group, seed]
                        assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream
                        assert partition == source['partition'] == anchor['partition'] == po['partition']
                        assert h(tid) == anchor['training_ids_hash'] and h(vid) == anchor['validation_ids_hash']
                        assert h(y[val]) == anchor['targets_hash'] and h(q[vid]) == anchor['past_quality_hash']
                        fitted, meta = cost_reader.fetch(anchor['checkpoint'])
                        fp, mp = poisson_reader.fetch(po['checkpoint'])
                        for cp, m, a, directory in ((fitted, meta, anchor, diagnostic.fitted_run.PUBLIC), (fp, mp, po, prior.PUBLIC)):
                            assert m['fit'] == a['training']
                            assert m['identity'] == dict(group=group, source=site, head_seed=seed, partition=partition,
                                checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(directory/'registration.json'))
                            assert api.forest.fingerprint(q[tid]) == m['fit']['past_quality_hash']
                        op, add, support, _, _, _ = prior.additive_control(state, add_reader, ad, source, partition, tid, vid, x[val], env[val], q[vid])
                        _, pp, sp, _ = api.diagnostic.model.positive.predict(state, fp, x[val], env[val], q[vid])
                        np.testing.assert_array_equal(sp, support)
                        fetched += sum(anchor[k]['bytes'] for k in ('checkpoint', 'poisson_checkpoint', 'additive_checkpoint'))
                        bounds = api.training_bounds(state, fitted, x[tr], env[tr], y[tr], q[tid],
                            data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                        train_raw, _, train_changed = api.predict_raw(state, fitted, bounds, x[tr], env[tr], q[tid])
                        known = np.isfinite(y[tr]).all(1)
                        np.testing.assert_array_equal(train_changed[known], 0)
                        np.testing.assert_array_equal(train_raw['cost'][known], train_raw['extended'][known])
                        train_hashes = {k: h(train_raw[k][known]) for k in ('cost', 'extended')}
                        raw, sp, changed = api.predict_raw(state, fitted, bounds, x[val], env[val], q[vid])
                        repeated, again_support, again_changed = api.predict_raw(state, fitted, bounds, x[val], env[val], q[vid])
                        for k in raw:
                            np.testing.assert_array_equal(raw[k], repeated[k])
                        np.testing.assert_array_equal(sp, support)
                        np.testing.assert_array_equal(sp, again_support)
                        np.testing.assert_array_equal(changed, again_changed)
                        pred = {k: api.forest.project_moments(v, env[val]) for k, v in raw.items()}
                        pred.update(additive=add, poisson=pp)
                        np.testing.assert_array_equal(pred['original'], op)
                        for k in api.CONTROLS:
                            assert h(pred[k]) == anchor['prediction_hashes'][k]
                        kw = dict(state=state, targets=y[val], envelope=env[val], moving=c['moving'][at][val],
                            support=support, sites=data['sites'][vid], recordings=data['recordings'][vid], frames=data['frames'][vid], ids=vid)
                        frozen, frozen_actions = api.diagnostic.model.evaluate(predictions={k: pred[k] for k in api.CONTROLS}, **kw)
                        assert frozen == anchor['result']
                        for k, act in frozen_actions.items():
                            assert h(act) == anchor['action_hashes'][k]
                        result, actions = api.evaluate(predictions=pred, **kw)
                        for k, act in actions.items():
                            checks += prior.prior.leaf.previous.check_scalars(result['policies'][k],
                                prior.prior.leaf.previous.independent.scalar_bounds(y[val], act, env[val]))
                        weights, _ = api.forest.core.weights(data['sites'][vid], data['recordings'][vid], data['frames'][vid], np.isfinite(y[val]).all(1))
                        masks = dict(all=np.ones(len(vid), bool), extended_selected=actions['extended'],
                            extended_unselected=~actions['extended'], any_quality_clipped=changed > 0, no_quality_clipped=changed == 0)
                        slices = {k: api.diagnostic.prior.error_slice(pred['cost'], pred['extended'], y[val], weights, state['preprocess'], m) for k, m in masks.items()}
                        raw_scores = {k: api.forest.signed_score(v, y[val], state['preprocess'], data['sites'][vid], data['recordings'][vid], data['frames'][vid]) for k, v in raw.items()}
                        np.testing.assert_allclose(slices['all']['global_weighted_MSE_change'], result['contrasts']['extended_minus_cost_signed_MSE'], atol=1e-9, rtol=1e-9)
                        row = dict(group=group, source=site, head_seed=seed, partition=partition, result=result,
                            slices=slices, raw_scores=raw_scores, result_source='fresh_run_train_identity_extension',
                            models_source='cached_verified', new_training=False, training_known_rows=int(known.sum()),
                            training_unknown_rows=int((~known).sum()), known_training_prediction_hashes=train_hashes,
                            validation_changed_rows=int((changed > 0).sum()), validation_rows=len(vid),
                            bounds_hashes={k: h(v) for k, v in bounds.items()},
                            checkpoint=anchor['checkpoint'], parent_checkpoint=source['checkpoint'],
                            additive_checkpoint=anchor['additive_checkpoint'], poisson_checkpoint=anchor['poisson_checkpoint'],
                            training_ids_hash=h(tid), validation_ids_hash=h(vid), targets_hash=h(y[val]), past_quality_hash=h(q[vid]),
                            prediction_hashes={k: h(v) for k, v in pred.items()}, action_hashes={k: h(v) for k, v in actions.items()},
                            registration_sha256=sha(PUBLIC/'registration.json'))
                        rows.append(row)
                        if args.phase != 'pilot':
                            path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                            if path.exists() and not (args.resume or args.phase == 'verify'):
                                raise RuntimeError('Explicit resume required')
                            once(path, row)
                            size += path.stat().st_size
                            assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(rows), seconds=time.monotonic()-start, fetched_bytes=fetched)
                        if args.phase == 'pilot': break
                    if args.phase == 'pilot': break
                if args.phase == 'pilot': break
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
            fetched_checkpoint_bytes=fetched, independent_scalar_checks=checks,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            native_architecture=platform.machine(), cpu_threads=cfg['cpu_threads'], num_workers=0,
            known_training_predictions_exact=True, exact_inference_replay=True, controls_exact_replay=True,
            new_training=False, local_numeric_cache=False, new_HPC_jobs=0)
        if args.phase == 'pilot':
            once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(rows) == 72
            once(PUBLIC/'summary.json', api.summarize(rows, cfg))
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            dest = PRIVATE/'additional_replays'/(str(time.time_ns())+'.json') if args.phase == 'verify' else PUBLIC/'complete.json'
            once(dest, runtime)
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__':
    main()
