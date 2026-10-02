"""Train a cost-aligned positive head against three immutable controls."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_positive_harm_diagnostic_v2 as diagnostic
from scripts import run_m3w_positive_harm_diagnostic as reader
from src.world_model import m3w_cost_aligned_positive_harm as api

prior, parent, sha, once = reader.prior, reader.parent, reader.sha, reader.once
NAME = 'european_cost_aligned_positive_harm_v1'
PUBLIC, PRIVATE = prior.PUBLIC.parent/NAME, prior.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
REMOTE = '/users/k24101830/m3w/'+NAME


def registration():
    assert diagnostic.registration() == json.loads((diagnostic.PUBLIC/'registration.json').read_text())
    v = json.loads((diagnostic.PUBLIC/'verification.json').read_text())
    for name in ('summary', 'complete'): assert sha(diagnostic.PUBLIC/(name+'.json')) == v[name+'_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_cost_aligned_positive_harm'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_cost_aligned_positive_harm.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        diagnostic_verification_sha256=sha(diagnostic.PUBLIC/'verification.json'),
        source_heads=72, independent_roles_read=False, new_neural_updates=0, new_tree_splits=0)


def storage(refs=None):
    code = r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_cost_aligned_positive_harm_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and shutil.disk_usage(r.parent).free>p['cap']
r.mkdir(exist_ok=True)
for name,txt in {'.owner.json':json.dumps(dict(experiment=p['name'],project='M3W'))+'\n',
                 'registration.json':p['registration'],'config.json':p['config']}.items():
    f=r/name
    if f.exists():assert f.read_text()==txt
    else:
        with f.open('x') as out:out.write(txt)
existing=[]
for f in sorted((r/'inputs').glob('*.npz')):
    existing.append(dict(group=f.stem,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
if p['refs'] is not None:assert sorted(existing,key=lambda x:x['group'])==sorted(p['refs'],key=lambda x:x['group'])
print(json.dumps(dict(existing=existing,owned_root_verified=True,personal_quota='unknown',
    remote_science_executed=False,m3w_jobs=[l for l in q.stdout.splitlines() if 'm3w' in l.lower()])))
'''
    cfg = json.loads(CONFIG.read_text())
    return prior.prior.leaf.previous.independent.read_remote(code, dict(root=REMOTE, name=NAME,
        cap=cfg['remote_checkpoint_cap_bytes'], registration=(PUBLIC/'registration.json').read_text(),
        config=CONFIG.read_text(), refs=refs))


class Stream(prior.Stream):
    def __init__(self):
        handoff = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(handoff/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((handoff/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.command = ssh+[shlex.join(['/usr/bin/python3', '-c', prior.prior.RECEIVER, REMOTE, NAME])]
        self.process = None


def beat(**kw):
    value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(value)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify']); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register': once(PUBLIC/'registration.json', reg); print('Registered squared cost control'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text()); parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify': assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic(); pilot = args.phase == 'pilot'
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = storage(); existing = {r['group']: r for r in preflight['existing']}
        if existing and not (args.resume or args.phase == 'verify'): raise RuntimeError('Existing checkpoint requires resume')
        beat(state='storage_verified', **{k: v for k, v in preflight.items() if k != 'existing'})
        poisson_rows = {}
        for ref in json.loads((prior.PUBLIC/'complete.json').read_text())['groups']:
            assert sha(ROOT/ref['path']) == ref['sha256']
            r = json.loads((ROOT/ref['path']).read_text()); poisson_rows[r['group'], r['head_seed']] = r
        docs, cal, old = parent.parent.docs(), parent.docs(), prior.prior_rows()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.prior.past_quality(data)
        assert provenance == json.loads((prior.prior.PUBLIC/'feature_provenance.json').read_text())
        rows = []; refs = []; cps = []; checks = size = cpbytes = fresh = reused = 0; fit_seconds = 0.
        with Stream() as stream, reader.PositiveReader() as poisson_reader, prior.previous.Reader() as add_reader:
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]; h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        source, ca, po, ad = docs[group, seed], cal[group, seed], poisson_rows[group, seed], old[group, seed]
                        path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                        assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                        assert h(tid) == po['training_ids_hash'] and h(vid) == po['validation_ids_hash']
                        assert h(y[val]) == po['targets_hash'] == ca['identity']['target_hash'] and h(q[vid]) == po['past_quality_hash']
                        if args.resume and path.exists() and args.phase != 'verify':
                            row = json.loads(path.read_text()); cp = row['checkpoint']
                            assert row['registration_sha256'] == sha(PUBLIC/'registration.json') and row['partition'] == partition
                            assert row['training_ids_hash'] == h(tid) and row['validation_ids_hash'] == h(vid)
                            assert row['targets_hash'] == h(y[val]) and row['parent_checkpoint'] == source['checkpoint']
                            assert existing[cp['group']] == cp
                            rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                            cps.append(cp); cpbytes += cp['bytes']; reused += 1
                            beat(state='cached_verified_resumed', groups=len(rows)); continue
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream and source['partition'] == partition == po['partition']
                        original, add, support, additive, _, _ = prior.additive_control(state, add_reader, ad,
                            source, partition, tid, vid, x[val], env[val], q[vid])
                        fitted_poisson, meta = poisson_reader.fetch(po['checkpoint'])
                        assert meta['fit'] == po['training']
                        assert meta['identity'] == dict(group=group, source=site, head_seed=seed, partition=partition,
                            checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(prior.PUBLIC/'registration.json'))
                        op, pp, sp, _ = api.positive.predict(state, fitted_poisson, x[val], env[val], q[vid])
                        np.testing.assert_array_equal(op, original); np.testing.assert_array_equal(sp, support)
                        control = dict(original=original, additive=add, positive=pp)
                        for k, v in control.items(): assert h(v) == po['prediction_hashes'][k]
                        kw = dict(state=state, targets=y[val], envelope=env[val], moving=c['moving'][at][val], support=support,
                            sites=data['sites'][vid], recordings=data['recordings'][vid], frames=data['frames'][vid], ids=vid)
                        old_result, old_actions = api.positive.evaluate(predictions=control, **kw); assert old_result == po['result']
                        for k, v in old_actions.items(): assert h(v) == po['action_hashes'][k]
                        before = time.monotonic(); beat(state='fitting_squared_cost', group=group, head_seed=seed)
                        fit_args = (state, x[tr], env[tr], y[tr], q[tid], data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                        fitted = api.fit(*fit_args, settings=cfg['fit_settings']); replay = api.fit(*fit_args, settings=cfg['fit_settings'])
                        for k in fitted:
                            if isinstance(fitted[k], np.ndarray): np.testing.assert_array_equal(fitted[k], replay[k])
                            else: assert fitted[k] == replay[k]
                        for k in ('quality_mean', 'quality_std'): np.testing.assert_array_equal(fitted[k], additive[k])
                        fit_seconds += time.monotonic()-before; fresh += 1
                        identity = dict(group=group, source=site, head_seed=seed, partition=partition,
                            checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(PUBLIC/'registration.json'))
                        payload = prior.prior.serialize(fitted, identity); assert payload == prior.prior.serialize(replay, identity)
                        cpbytes += len(payload); assert cpbytes <= cfg['remote_checkpoint_cap_bytes']
                        cp = stream.send(group+'_head'+str(seed)+'_cost', payload); cps.append(cp)
                        op, new, sp, details = api.predict(state, fitted, x[val], env[val], q[vid])
                        np.testing.assert_array_equal(op, original); np.testing.assert_array_equal(sp, support)
                        with np.load(io.BytesIO(payload), allow_pickle=False) as z: loaded = {k: z[k] for k in z.files if k != 'meta_json'}
                        repeated = api.predict(state, loaded, x[val], env[val], q[vid])
                        for a, b in zip((op, new, sp), repeated[:3]): np.testing.assert_array_equal(a, b)
                        assert details == repeated[3]
                        pred = dict(original=original, additive=add, poisson=pp, cost=new)
                        result, actions = api.evaluate(predictions=pred, **kw)
                        again, _ = api.evaluate(predictions=pred, **kw); assert again == result
                        for k, action in actions.items():
                            checks += prior.prior.leaf.previous.check_scalars(result['policies'][k],
                                prior.prior.leaf.previous.independent.scalar_bounds(y[val], action, env[val]))
                        row = dict(group=group, source=site, head_seed=seed, partition=partition, result=result,
                            result_source='fresh_run_train_only_squared_cost', controls_source='cached_verified',
                            training={k: v for k, v in fitted.items() if not isinstance(v, np.ndarray)}, diagnostic=details,
                            checkpoint=cp, parent_checkpoint=source['checkpoint'], poisson_checkpoint=po['checkpoint'],
                            additive_checkpoint=po['additive_checkpoint'], prediction_hashes={k: h(v) for k, v in pred.items()},
                            action_hashes={k: h(v) for k, v in actions.items()}, training_ids_hash=h(tid), validation_ids_hash=h(vid),
                            targets_hash=h(y[val]), past_quality_hash=h(q[vid]), registration_sha256=sha(PUBLIC/'registration.json'))
                        rows.append(row)
                        if not pilot:
                            once(path, row); size += path.stat().st_size; assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(rows), fit_refit_seconds=fit_seconds,
                            checkpoint_bytes=cpbytes, seconds=time.monotonic()-start)
                        if pilot: break
                    if pilot: break
                if pilot: break
        verified = storage(cps)
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs, fit_refit_seconds=fit_seconds,
            fresh_fits_this_invocation=fresh, cached_verified_resumed_groups=reused, scalar_checks=checks,
            checkpoint_count=len(cps), checkpoint_bytes=cpbytes, owned_remote_weights_verified=len(verified['existing']),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            local_numeric_cache=False, new_HPC_jobs=0, exact_fit_replay=True, exact_inference_replay=True)
        if pilot: once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(rows) == 72; summary = api.summarize(rows, cfg)
            once(PUBLIC/'summary.json', summary); once(PUBLIC/'checkpoint_manifest.json', dict(checkpoints=cps))
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            dest = PRIVATE/'additional_replays'/(str(time.time_ns())+'.json') if args.phase == 'verify' else PUBLIC/'complete.json'
            once(dest, runtime); print(json.dumps(summary, indent=2))
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__': main()
