"""Train a fixed positive conditional harm head with frozen additive controls."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_quality_components as previous
from src.world_model import m3w_positive_harm as api

prior, parent = previous.prior, previous.parent
sha, once = previous.sha, previous.once
NAME = 'european_positive_harm_v1'
PUBLIC, PRIVATE = previous.PUBLIC.parent/NAME, previous.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
REMOTE = '/users/k24101830/m3w/'+NAME


def registration():
    assert previous.registration() == json.loads((previous.PUBLIC/'registration.json').read_text())
    v = json.loads((previous.PUBLIC/'verification.json').read_text())
    assert sha(previous.PUBLIC/'summary.json') == v['summary_sha256']
    assert sha(previous.PUBLIC/'complete.json') == v['complete_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_positive_harm'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_positive_harm.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
                component_verification_sha256=sha(previous.PUBLIC/'verification.json'),
                source_heads=72, new_conditional_harm_fits=72, cached_additive_controls=72,
                independent_roles_read=False, new_neural_updates=0)


def storage(refs=None):
    code = r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_positive_harm_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0
assert shutil.disk_usage(r.parent).free>p['cap']
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
if p['refs'] is not None:
    assert sorted(existing,key=lambda v:v['group'])==sorted(p['refs'],key=lambda v:v['group'])
print(json.dumps(dict(existing=existing,owned_root_verified=True,personal_quota='unknown',
    remote_science_executed=False,m3w_jobs=[l for l in q.stdout.splitlines() if 'm3w' in l.lower()])))
'''
    cfg = json.loads(CONFIG.read_text())
    return prior.leaf.previous.independent.read_remote(code, dict(root=REMOTE, name=NAME,
        cap=cfg['remote_checkpoint_cap_bytes'], registration=(PUBLIC/'registration.json').read_text(),
        config=CONFIG.read_text(), refs=refs))


class Stream(prior.PacketStream):
    def __init__(self):
        handoff = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(handoff/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((handoff/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.command = ssh+[shlex.join(['/usr/bin/python3', '-c', prior.RECEIVER, REMOTE, NAME])]
        self.process = None


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def prior_rows():
    rows = {}
    for ref in json.loads((prior.PUBLIC/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text()); rows[r['group'], r['head_seed']] = r
    return rows


def additive_control(state, reader, row, source, partition, tid, vid, x, env, q):
    h = parent.base.inter.array_hash
    cp = row['checkpoints']['quality']; fitted, meta = reader.fetch(cp)
    identity = meta['identity']
    assert identity['group'] == row['group'] and identity['head_seed'] == row['head_seed']
    assert identity['arm'] == 'quality' and identity['partition'] == partition
    assert identity['checkpoint'] == source['checkpoint'] and identity['training_ids_hash'] == h(tid)
    assert identity['registration_sha256'] == sha(prior.PUBLIC/'registration.json')
    assert meta['fit'] == row['training']['quality'] and h(vid) == row['validation_ids_hash']
    assert h(q) == row['past_quality_hash']
    raw, delta, support = previous.api.raw_predictions(state, fitted, x, env, q)
    old = api.base.forest.project_moments(raw, env)
    full = api.base.forest.project_moments(np.maximum(raw+delta, 0), env)
    assert h(old) == row['prediction_hashes']['original'] and h(full) == row['prediction_hashes']['quality']
    change = np.zeros_like(delta); change[:, api.HARM] = delta[:, api.HARM]
    new = api.base.forest.project_moments(np.maximum(raw+change, 0), env)
    return old, new, support, fitted, full, dict(
        negative_raw_coordinates=int((raw+change < 0).sum()),
        projected_benefit_changed=int((new[:, 0] != old[:, 0]).sum()),
        zero_predicted_harm=int((new[:, 1] == 0).sum()), zero_predicted_easy_harm=int((new[:, 4] == 0).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg); print('Registered positive harm training'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify': assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic(); pilot = args.phase == 'pilot'
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = storage()
        beat(state='storage_verified', **{k: v for k, v in preflight.items() if k != 'existing'})
        if preflight['existing'] and not (args.resume or args.phase == 'verify'):
            raise RuntimeError('Existing checkpoint requires --resume')
        existing = {v['group']: v for v in preflight['existing']}
        docs, cal, old = parent.parent.docs(), parent.docs(), prior_rows()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.past_quality(data)
        assert provenance == json.loads((prior.PUBLIC/'feature_provenance.json').read_text())
        rows = []; refs = []; cps = []; checks = size = cpbytes = fresh = reused = 0
        fit_seconds = infer_seconds = 0.
        with Stream() as stream, previous.Reader() as reader:
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]; h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        source, ca, oldrow = docs[group, seed], cal[group, seed], old[group, seed]
                        assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                        path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                        if args.resume and path.exists() and args.phase != 'verify':
                            row = json.loads(path.read_text()); cp = row['checkpoint']
                            assert row['registration_sha256'] == sha(PUBLIC/'registration.json')
                            assert row['partition'] == partition and row['training_ids_hash'] == h(tid)
                            assert row['validation_ids_hash'] == h(vid) and row['targets_hash'] == h(y[val])
                            assert row['parent_checkpoint'] == source['checkpoint'] and existing[cp['group']] == cp
                            rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                            cps.append(cp); cpbytes += cp['bytes']; reused += 1
                            beat(state='cached_verified_group_resumed', groups=len(rows)); continue
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream and source['partition'] == partition
                        oldpred, add, support, additive, full, ad = additive_control(
                            state, reader, oldrow, source, partition, tid, vid, x[val], env[val], q[vid])
                        assert h(vid) == ca['identity']['validation_ids_hash']
                        assert h(y[val]) == ca['identity']['target_hash'] and h(env[val]) == ca['identity']['envelope_hash']
                        assert h(api.base.eligible(full, c['moving'][at][val], support)) == oldrow['action_hashes']['quality']
                        before = time.monotonic(); beat(state='fitting_positive_harm', group=group, head_seed=seed)
                        args_fit = (state, x[tr], env[tr], y[tr], q[tid], data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                        fitted = api.fit(*args_fit, settings=cfg['fit_settings'])
                        replay = api.fit(*args_fit, settings=cfg['fit_settings'])
                        for k in fitted:
                            if isinstance(fitted[k], np.ndarray): np.testing.assert_array_equal(fitted[k], replay[k])
                            else: assert fitted[k] == replay[k]
                        for k in ('quality_mean', 'quality_std'): np.testing.assert_array_equal(fitted[k], additive[k])
                        fit_seconds += time.monotonic()-before; fresh += 1
                        identity = dict(group=group, source=site, head_seed=seed, partition=partition,
                            checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(PUBLIC/'registration.json'))
                        payload = prior.serialize(fitted, identity); assert payload == prior.serialize(replay, identity)
                        cpbytes += len(payload); assert cpbytes <= cfg['remote_checkpoint_cap_bytes']
                        cp = stream.send(group+'_head'+str(seed)+'_positive', payload); cps.append(cp)
                        before = time.monotonic()
                        original, positive, support2, diag = api.predict(state, fitted, x[val], env[val], q[vid])
                        np.testing.assert_array_equal(oldpred, original); np.testing.assert_array_equal(support, support2)
                        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                            loaded = {k: z[k] for k in z.files if k != 'meta_json'}
                        again = api.predict(state, loaded, x[val], env[val], q[vid])
                        for a, b in zip((original, positive, support), again[:3]): np.testing.assert_array_equal(a, b)
                        assert diag == again[3]
                        pred = dict(original=original, additive=add, positive=positive)
                        kw = dict(state=state, predictions=pred, targets=y[val], envelope=env[val],
                            moving=c['moving'][at][val], support=support, sites=data['sites'][vid],
                            recordings=data['recordings'][vid], frames=data['frames'][vid], ids=vid)
                        result, actions = api.evaluate(**kw); repeated, _ = api.evaluate(**kw); assert result == repeated
                        assert h(actions['original']) == source['validation']['action_hashes']['forest']
                        for name, a in actions.items():
                            checks += prior.leaf.previous.check_scalars(result['policies'][name], prior.leaf.previous.independent.scalar_bounds(y[val], a, env[val]))
                        checks += prior.leaf.previous.check_scalars(result['policies']['original'], source['validation']['completion_screen'])
                        infer_seconds += time.monotonic()-before
                        row = dict(group=group, source=site, head_seed=seed, partition=partition, result=result,
                            result_source='fresh_run_train_only_positive_harm', controls_source='cached_verified',
                            training={k: v for k, v in fitted.items() if not isinstance(v, np.ndarray)},
                            checkpoint=cp, parent_checkpoint=source['checkpoint'], additive_checkpoint=oldrow['checkpoints']['quality'],
                            diagnostic=dict(positive=diag, additive=ad), prediction_hashes={k: h(v) for k, v in pred.items()},
                            action_hashes={k: h(v) for k, v in actions.items()}, training_ids_hash=h(tid), validation_ids_hash=h(vid),
                            targets_hash=h(y[val]), past_quality_hash=h(q[vid]), registration_sha256=sha(PUBLIC/'registration.json'))
                        rows.append(row)
                        if not pilot:
                            if path.exists() and args.phase != 'verify': raise RuntimeError('Existing group requires --resume')
                            once(path, row); size += path.stat().st_size; assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(rows), fit_refit_seconds=fit_seconds,
                            inference_readout_seconds=infer_seconds, checkpoint_bytes=cpbytes, seconds=time.monotonic()-start)
                        if pilot: break
                    if pilot: break
                if pilot: break
        verified = storage(cps)
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
            fit_refit_seconds=fit_seconds, inference_readout_seconds=infer_seconds,
            fresh_fits_this_invocation=fresh, cached_verified_resumed_groups=reused, scalar_checks=checks,
            checkpoint_bytes=cpbytes, checkpoint_count=len(cps), owned_remote_weights_verified=len(verified['existing']),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            available_disk_bytes=shutil.disk_usage(ROOT).free, local_numeric_cache=False, new_HPC_jobs=0,
            exact_fit_replay=True, exact_inference_replay=True, exact_readout_replay=True,
            native_architecture=platform.machine(), cpu_threads=cfg['cpu_threads'], num_workers=0)
        if pilot:
            runtime['first_head_projection_not_runtime_bound_seconds'] = (fit_seconds+infer_seconds)*72
            once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(rows) == cfg['source_heads']
            summary = api.summarize(rows, cfg); once(PUBLIC/'summary.json', summary)
            once(PUBLIC/'checkpoint_manifest.json', dict(remote_root=REMOTE, checkpoints=cps))
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            if args.phase == 'verify': once(PRIVATE/'additional_replays'/(str(time.time_ns())+'.json'), runtime)
            else: once(PUBLIC/'complete.json', runtime)
            print(json.dumps(summary, indent=2))
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__': main()
