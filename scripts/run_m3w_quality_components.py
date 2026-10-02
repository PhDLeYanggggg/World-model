"""Read frozen quality checkpoints in memory and diagnose five cost components."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import select
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
from scripts import run_m3w_past_quality_auxiliary as prior
from src.world_model import m3w_quality_component_diagnostic as api

parent, sha, once = prior.parent, prior.sha, prior.once
NAME = 'european_quality_components_v1'
PUBLIC, PRIVATE = prior.PUBLIC.parent/NAME, prior.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')

SERVER = r'''
import hashlib,json,pathlib,re,sys
root=pathlib.Path('/users/k24101830/m3w/european_past_quality_auxiliary_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_past_quality_auxiliary_v1'
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin.buffer:
    assert len(line)<=4096
    name=json.loads(line)['group']
    assert re.fullmatch(r'[A-Za-z0-9_-]+_quality',name)
    p=root/'inputs'/(name+'.npz');n=p.stat().st_size
    assert 0<n<=32*2**20
    raw=p.read_bytes()
    print(json.dumps(dict(group=name,bytes=n,sha256=hashlib.sha256(raw).hexdigest())),flush=True)
    sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
'''


def read_bytes(stream, count):
    if not 0 <= count <= 32*2**20:
        raise ValueError('Bounded immutable checkpoint required')
    parts = []
    while count:
        if hasattr(stream, 'fileno'):
            try:
                fd = stream.fileno()
            except io.UnsupportedOperation:
                fd = None
            if fd is not None and not select.select([fd], [], [], 120)[0]:
                raise TimeoutError('Read-only transport timeout; preserve all prior work')
        chunk = stream.read(min(count, 2**20))
        if not chunk:
            raise EOFError('Truncated immutable checkpoint')
        parts.append(chunk); count -= len(chunk)
    return b''.join(parts)


class Reader:
    def __enter__(self):
        handoff = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(handoff/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((handoff/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.p = subprocess.Popen(ssh+[shlex.join(['/usr/bin/python3', '-c', SERVER])],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        try:
            assert self.line() == {'ready': True}
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def line(self):
        value = bytearray()
        while not value.endswith(b'\n'):
            if len(value) >= 4096:
                raise ValueError('Oversized checkpoint header')
            value.extend(read_bytes(self.p.stdout, 1))
        return json.loads(value)

    def fetch(self, ref):
        self.p.stdin.write((json.dumps(dict(group=ref['group']))+'\n').encode())
        self.p.stdin.flush()
        assert self.line() == ref
        raw = read_bytes(self.p.stdout, ref['bytes'])
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            fit = {k: z[k] for k in z.files if k != 'meta_json'}
            meta = json.loads(str(z['meta_json']))
        return fit, meta

    def __exit__(self, *args):
        p = self.p
        if not p.stdin.closed:
            p.stdin.close()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # This is only our read-only transport, not a scheduler/science job.
            p.terminate(); p.wait(timeout=10)
        error = p.stderr.read().decode(errors='replace')[-1000:]
        p.stdout.close(); p.stderr.close()
        if args[0] is None and p.returncode:
            raise RuntimeError('Read-only checkpoint transport failed: '+error)


def registration():
    assert prior.registration() == json.loads((prior.PUBLIC/'registration.json').read_text())
    v = json.loads((prior.PUBLIC/'verification.json').read_text())
    assert sha(prior.PUBLIC/'complete.json') == v['complete_sha256']
    assert sha(prior.PUBLIC/'summary.json') == v['summary_sha256']
    assert sha(prior.PUBLIC/'checkpoint_manifest.json') == v['checkpoint_manifest_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_quality_components'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_quality_components.py',
              ROOT/'tests/test_m3w_quality_component_runner.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
                parent_verification_sha256=sha(prior.PUBLIC/'verification.json'),
                independent_roles_read=False, training=False,
                source_heads=72, components=list(api.COMPONENTS))


def summarize(rows, cfg):
    out = dict(groups=len(rows), training=False, independent_confirmation=False,
               deployment_changed=False, policy_selection=False, arms={})
    for name in rows[0]['result']['arms']:
        values = [g['result']['arms'][name] for g in rows]
        def ci(field):
            return prior.api.interval([(g['source'], g['result']['arms'][name][field]) for g in rows],
                                      cfg['bootstrap_draws'], cfg['bootstrap_seed'])
        risk = [v['full']['easy_selected_risk_upper'] for v in values
                if v['full']['easy_selected_risk_upper'] is not None]
        out['arms'][name] = dict(
            selected=sum(v['full']['selected_count'] for v in values),
            unknown_selected=sum(v['full']['selected_unknown'] for v in values),
            complete_support=sum(v['full']['finite_completion_supported'] for v in values),
            defined_easy_risk=len(risk), upper_violations=sum(v > .02+1e-12 for v in risk),
            worst_easy_risk_upper=max(risk) if risk else None,
            known_label_violations=sum(v['full']['selected_known_easy_harm_mass']/
                v['full']['selected_known_easy_reference_mass'] > .02+1e-12 for v in values
                if v['full']['selected_known_easy_reference_mass'] > 0),
            added=sum(v['cohorts']['added']['selected'] for v in values),
            removed=sum(v['cohorts']['removed']['selected'] for v in values),
            full_utility_difference_percent=ci('full_utility_difference_percent'),
            matched_utility_difference_percent=ci('matched_utility_difference_percent'))
    out['projection_added_actions'] = sum(g['result']['quality_projection_added_actions'] for g in rows)
    out['projection_removed_actions'] = sum(g['result']['quality_projection_removed_actions'] for g in rows)
    return out


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
    reg, cfg = registration(), json.loads(CONFIG.read_text())
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg); print('Registered frozen component diagnosis'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify':
        assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic(); refs = []; groups = []; checks = size = fetched = 0
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docs, cal = parent.parent.docs(), parent.docs()
        old_complete = json.loads((prior.PUBLIC/'complete.json').read_text())
        old = {}
        for ref in old_complete['groups']:
            assert sha(ROOT/ref['path']) == ref['sha256']
            row = json.loads((ROOT/ref['path']).read_text()); old[row['group'], row['head_seed']] = row
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.past_quality(data)
        assert provenance == json.loads((prior.PUBLIC/'feature_provenance.json').read_text())
        with Reader() as reader:
            beat(state='owned_read_only_transport_connected', new_HPC_jobs=0)
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(
                        data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]; h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        beat(state='frozen_inference', group=group, head_seed=seed)
                        prior_row, source, ca = old[group, seed], docs[group, seed], cal[group, seed]
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream and source['partition'] == partition
                        cp = prior_row['checkpoints']['quality']; fit, meta = reader.fetch(cp); fetched += cp['bytes']
                        identity = meta['identity']
                        assert identity['group'] == group and identity['head_seed'] == seed and identity['arm'] == 'quality'
                        assert identity['partition'] == partition and identity['checkpoint'] == source['checkpoint']
                        assert identity['training_ids_hash'] == h(tid)
                        assert identity['registration_sha256'] == sha(prior.PUBLIC/'registration.json')
                        assert meta['fit'] == prior_row['training']['quality']
                        assert h(vid) == prior_row['validation_ids_hash'] == ca['identity']['validation_ids_hash']
                        assert h(y[val]) == ca['identity']['target_hash'] and h(env[val]) == ca['identity']['envelope_hash']
                        assert h(q[vid]) == prior_row['past_quality_hash']
                        raw, delta, support = api.raw_predictions(state, fit, x[val], env[val], q[vid])
                        repeat = api.raw_predictions(state, fit, x[val], env[val], q[vid])
                        for a, b in zip((raw, delta, support), repeat):
                            np.testing.assert_array_equal(a, b)
                        predictions = api.variants(raw, delta, env[val])
                        for name in ('original', 'quality'):
                            assert h(predictions[name]) == prior_row['prediction_hashes'][name]
                        result = api.evaluate(predictions, y[val], env[val], c['moving'][at][val], support,
                                              data['recordings'][vid], data['frames'][vid], vid)
                        repeated = api.evaluate(predictions, y[val], env[val], c['moving'][at][val], support,
                                                data['recordings'][vid], data['frames'][vid], vid)
                        assert result == repeated
                        for name in ('original', 'quality'):
                            checks += prior.leaf.previous.check_scalars(result['arms'][name]['full'],
                                                                       prior_row['result']['policies'][name])
                            expected = api.quality.eligible(predictions[name], c['moving'][at][val], support)
                            assert h(expected) == prior_row['action_hashes'][name]
                        checks += prior.leaf.previous.check_scalars(result['arms']['quality']['original_matched'],
                                                                   prior_row['result']['policies']['original_matched_quality'])
                        checks += prior.leaf.previous.check_scalars(result['arms']['quality']['variant_matched'],
                                                                   prior_row['result']['policies']['quality_matched_original'])
                        row = dict(group=group, source=site, head_seed=seed, result=result,
                                   result_source='fresh_run_frozen_component_inference_no_training',
                                   parent_checkpoint=cp, partition=partition,
                                   validation_ids_hash=h(vid), targets_hash=h(y[val]),
                                   registration_sha256=sha(PUBLIC/'registration.json'))
                        groups.append(row)
                        if args.phase != 'pilot':
                            path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                            if path.exists() and not (args.resume or args.phase == 'verify'):
                                raise RuntimeError('Existing reports require resume')
                            once(path, row); size += path.stat().st_size
                            assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(groups), seconds=time.monotonic()-start,
                             fetched_checkpoint_bytes=fetched, parent_scalar_checks=checks)
                        if args.phase == 'pilot': break
                    if args.phase == 'pilot': break
                if args.phase == 'pilot': break
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
                       peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                       fetched_checkpoint_bytes=fetched, parent_scalar_checks=checks,
                       exact_inference_replay=True, exact_readout_replay=True,
                       new_training=False, local_numeric_cache=False, new_HPC_jobs=0,
                       native_architecture=platform.machine(), cpu_threads=4, num_workers=0)
        if args.phase == 'pilot':
            once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(groups) == 72
            summary = summarize(groups, cfg); once(PUBLIC/'summary.json', summary)
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            if args.phase == 'verify':
                once(PRIVATE/'additional_replays'/(str(time.time_ns())+'.json'), runtime)
            else:
                once(PUBLIC/'complete.json', runtime)
            print(json.dumps(summary, indent=2), flush=True)
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__':
    main()
