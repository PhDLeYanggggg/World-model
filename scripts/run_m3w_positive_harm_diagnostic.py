"""Replay frozen cost heads and decompose the failed positive-harm loss tradeoff."""
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
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_positive_harm as prior
from src.world_model import m3w_positive_harm_diagnostic as api

parent, sha, once = prior.parent, prior.sha, prior.once
NAME = 'european_positive_harm_diagnostic_v1'
PUBLIC, PRIVATE = prior.PUBLIC.parent/NAME, prior.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')

SERVER = r'''
import hashlib,json,pathlib,re,sys
root=pathlib.Path('/users/k24101830/m3w/european_positive_harm_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_positive_harm_v1'
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin.buffer:
    assert len(line)<=4096
    name=json.loads(line)['group'];assert re.fullmatch(r'[A-Za-z0-9_-]+_positive',name)
    path=root/'inputs'/(name+'.npz');raw=path.read_bytes();assert 0<len(raw)<=32*2**20
    print(json.dumps(dict(group=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())),flush=True)
    sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
'''


class PositiveReader(prior.previous.Reader):
    def __enter__(self):
        h = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(h/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((h/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.p = subprocess.Popen(ssh+[shlex.join(['/usr/bin/python3', '-c', SERVER])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        try: assert self.line() == {'ready': True}
        except BaseException:
            self.__exit__(*sys.exc_info()); raise
        return self


def registration():
    assert prior.registration() == json.loads((prior.PUBLIC/'registration.json').read_text())
    v = json.loads((prior.PUBLIC/'verification.json').read_text())
    for name in ('summary', 'complete', 'checkpoint_manifest'):
        assert sha(prior.PUBLIC/(name+'.json')) == v[name+'_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_positive_harm_diagnostic'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_positive_harm_diagnostic.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        parent_verification_sha256=sha(prior.PUBLIC/'verification.json'), source_heads=72,
        training=False, policy_selection=False, independent_roles_read=False)


def beat(**kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(r)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def summarize(rows, cfg):
    out = dict(groups=len(rows), source_localities=len({r['source'] for r in rows}), new_training=False,
        policy_selection=False, independent_confirmation=False, deployment_changed=False)
    def reduce(section, slice_name=None):
        get = lambda row: row['diagnosis'][section] if slice_name is None else row['diagnosis'][section][slice_name]
        result = dict(rows=sum(get(r)['rows'] for r in rows), unknown_rows=sum(get(r)['unknown_rows'] for r in rows))
        for key in ('known_query_weight_mass', 'global_weighted_MSE_change', 'conditional_MSE_change'):
            result[key] = prior.api.base.interval([(r['source'], get(r)[key]) for r in rows],
                                                   cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
        for field, names in (('moment_contributions', api.MOMENTS), ('score_contributions', api.SCORES)):
            result[field] = {name: prior.api.base.interval([(r['source'], get(r)[field][j]) for r in rows],
                cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for j, name in enumerate(names)}
        return result
    for section in ('cohorts', 'strata', 'projection_effects'):
        out[section] = {key: reduce(section, key) for key in rows[0]['diagnosis'][section]}
    for section in ('raw_error_change', 'additive_error_change'): out[section] = reduce(section)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify']); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register': once(PUBLIC/'registration.json', reg); print('Registered frozen cost diagnosis'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify': assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic(); pilot = args.phase == 'pilot'
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_rows = {}
        for ref in json.loads((prior.PUBLIC/'complete.json').read_text())['groups']:
            assert sha(ROOT/ref['path']) == ref['sha256']
            row = json.loads((ROOT/ref['path']).read_text()); source_rows[row['group'], row['head_seed']] = row
        docs, cal, old = parent.parent.docs(), parent.docs(), prior.prior_rows()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.prior.past_quality(data)
        assert provenance == json.loads((prior.prior.PUBLIC/'feature_provenance.json').read_text())
        rows = []; refs = []; checks = fetched = size = 0
        with PositiveReader() as pos_reader, prior.previous.Reader() as add_reader:
            beat(state='owned_read_only_checkpoint_streams_connected', new_HPC_jobs=0)
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]; h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        beat(state='frozen_inference_and_training_support', group=group, head_seed=seed)
                        source, ca, trained, addrow = docs[group, seed], cal[group, seed], source_rows[group, seed], old[group, seed]
                        assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream and source['partition'] == partition == trained['partition']
                        assert h(tid) == trained['training_ids_hash'] and h(vid) == trained['validation_ids_hash']
                        assert h(y[val]) == trained['targets_hash'] == ca['identity']['target_hash']
                        assert h(q[vid]) == trained['past_quality_hash']
                        fitted, meta = pos_reader.fetch(trained['checkpoint'])
                        assert meta['fit'] == trained['training']
                        assert meta['identity'] == dict(group=group, source=site, head_seed=seed, partition=partition,
                            checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(prior.PUBLIC/'registration.json'))
                        op, ap, support, additive, _, _ = prior.additive_control(state, add_reader, addrow,
                            source, partition, tid, vid, x[val], env[val], q[vid])
                        fetched += trained['checkpoint']['bytes']+trained['additive_checkpoint']['bytes']
                        table = api.training_support(state, x[tr], env[tr], y[tr], data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                        ro, rp, sup, descriptor = api.raw_positive(state, fitted, x[val], env[val], q[vid], table)
                        repeated = api.raw_positive(state, fitted, x[val], env[val], q[vid], table)
                        for a, b in zip((ro, rp, sup, descriptor), repeated): np.testing.assert_array_equal(a, b)
                        ro2, delta, _ = prior.previous.api.raw_predictions(state, additive, x[val], env[val], q[vid])
                        np.testing.assert_array_equal(ro, ro2); np.testing.assert_array_equal(support, sup)
                        raw_add = ro.copy(); raw_add[:, (1, 4)] += delta[:, (1, 4)]
                        raw = dict(original=ro, positive=rp, additive=raw_add)
                        pred = dict(original=op, positive=api.forest.project_moments(rp, env[val]), additive=ap)
                        for name in pred: assert h(pred[name]) == trained['prediction_hashes'][name]
                        kw = dict(state=state, predictions=pred, targets=y[val], envelope=env[val],
                            moving=c['moving'][at][val], support=support, sites=data['sites'][vid],
                            recordings=data['recordings'][vid], frames=data['frames'][vid], ids=vid)
                        result, actions = prior.api.evaluate(**kw); assert result == trained['result']
                        for name, action in actions.items():
                            assert h(action) == trained['action_hashes'][name]
                            checks += prior.prior.leaf.previous.check_scalars(result['policies'][name], prior.prior.leaf.previous.independent.scalar_bounds(y[val], action, env[val]))
                        args_diag = (state, raw, pred, y[val], env[val], actions, data['sites'][vid], data['recordings'][vid], data['frames'][vid], descriptor)
                        diagnosis = api.diagnose(*args_diag); assert diagnosis == api.diagnose(*args_diag)
                        np.testing.assert_allclose(diagnosis['cohorts']['all']['global_weighted_MSE_change'],
                            result['contrasts']['positive_minus_original_signed_MSE'], atol=1e-9, rtol=1e-9)
                        row = dict(group=group, source=site, head_seed=seed, diagnosis=diagnosis,
                            result_source='fresh_run_frozen_inference_error_decomposition', models_source='cached_verified',
                            checkpoint=trained['checkpoint'], additive_checkpoint=trained['additive_checkpoint'],
                            training_ids_hash=h(tid), validation_ids_hash=h(vid), targets_hash=h(y[val]),
                            registration_sha256=sha(PUBLIC/'registration.json'))
                        rows.append(row)
                        if not pilot:
                            path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                            if path.exists() and not (args.resume or args.phase == 'verify'): raise RuntimeError('Explicit resume required')
                            once(path, row); size += path.stat().st_size; assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(rows), seconds=time.monotonic()-start, fetched_bytes=fetched, scalar_checks=checks)
                        if pilot: break
                    if pilot: break
                if pilot: break
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
            fetched_checkpoint_bytes=fetched, independent_scalar_checks=checks,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            native_architecture=platform.machine(), cpu_threads=cfg['cpu_threads'], num_workers=0,
            exact_raw_inference_replay=True, exact_readout_replay=True, new_training=False,
            local_numeric_cache=False, new_HPC_jobs=0)
        if pilot: once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(rows) == cfg['source_heads']
            summary = summarize(rows, cfg); once(PUBLIC/'summary.json', summary)
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            if args.phase == 'verify': once(PRIVATE/'additional_replays'/(str(time.time_ns())+'.json'), runtime)
            else: once(PUBLIC/'complete.json', runtime)
            print(json.dumps(dict(summary_sha256=runtime['summary_sha256'], all=summary['cohorts']['all']), indent=2))
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__': main()
