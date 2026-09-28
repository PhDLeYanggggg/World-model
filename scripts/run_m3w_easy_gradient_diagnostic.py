"""Read-only fitting-source gradients on allocated CREATE CPUs, with group resume."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_m3w_easy_hurdle_portable import api, digest, unpack, np, torch
from src.world_model.m3w_easy_gradient_diagnostic import gradients, fitting_signal


def write_once(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text() == raw, 'Immutable diagnostic differs'
    else:
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(raw); os.replace(tmp, path)


def group_readout(parent, entry, references, cfg):
    name = entry['group']; path = parent/'inputs'/(name+'.npz')
    assert digest(path) == entry['sha256']
    a, pr, identity = unpack(path)
    roles = identity['roles']
    assert set(a['sites']) == set(roles['training_sites'])
    for key in ('producer_sites', 'controller_sites', 'held_sites'):
        assert not set(a['sites']) & set(roles[key])
    known = np.isfinite(a['y']).all(1)
    assert np.array_equal(known, pr['known'])
    z, env, norm = api.preprocess(a['x'], a['u'], a['env'], pr)
    groups, sources, _ = api.sampling.query_groups(a['sites'], a['recordings'], a['frames'], known)
    rng = torch.Generator().manual_seed(identity['seed']+cfg['sampler_seed_offset'])
    batches = [api.sampling.draw_queries(groups, sources, cfg['queries_per_batch'], rng)
               for _ in range(cfg['batches_per_group'])]
    sample_hash = hashlib.sha256(b''.join(ix.tobytes()+seg.tobytes()+qids.tobytes() for ix, seg, qids in batches)).hexdigest()
    states = {}
    rows = []
    for arm in cfg['arms']:
        ref = references[(name, arm)]; checkpoint = parent/ref['path']
        assert digest(checkpoint) == ref['sha256']
        state = api.head.read_checkpoint(checkpoint); states[arm] = state
        assert state['identity'] == identity and state['step'] == 2000
        api.sampling.exact(norm, state['norm'])
        for point in cfg['states']:
            model = api.initialize(z.shape[1], state['settings']['width'], state['seed'])
            model.load_state_dict(state['initial_model'] if point == 'initial' else state['model'])
            model.eval()
            for b, (ix, seg, qids) in enumerate(batches):
                assert known[ix].all()
                row = gradients(model, z[ix], env[ix], torch.from_numpy(a['y'][ix].astype(np.float32)),
                                torch.from_numpy(seg), cfg['queries_per_batch'])
                rows.append(dict(arm=arm, point=point, batch=b, sampled_rows=len(ix), **row))
        assert digest(checkpoint) == ref['sha256']
    api.assert_matched(states['marginal'], states['supervised'])
    for b in range(cfg['batches_per_group']):
        first = [r for r in rows if r['point'] == 'initial' and r['batch'] == b]
        assert first[0]['losses'] == first[1]['losses'] and first[0]['gradients'] == first[1]['gradients']
    return dict(group=name, identity=identity, packet_sha256=entry['sha256'],
        checkpoint_sha256={arm: references[(name, arm)]['sha256'] for arm in cfg['arms']},
        sample_hash=sample_hash, rows=rows,
        fitting_signal=fitting_signal(a['y'], groups, dict(zip(sorted(set(a['sites'])), sources))),
        known_rows=int(known.sum()), unknown_rows_excluded=int((~known).sum()),
        parameter_updates=0, held_outcomes_used=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True); p.add_argument('--parent-home', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--replay', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Diagnostic executes on allocated node, not login node')
    home, parent = Path(a.home), Path(a.parent_home)
    cfg = json.loads((home/'config.json').read_text()); registration = json.loads((home/'registration.json').read_text())
    assert cfg['groups'] == 108 and cfg['parameter_updates'] == 0 and cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('held_outcomes_used', 'independent_roles_read', 'threshold_search', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(cfg['interop_threads'])
    for relative, sha in registration['code_bindings'].items():
        assert digest(ROOT/relative) == sha
    assert digest(parent/'input_manifest.json') == cfg['parent_manifest_sha256']
    assert digest(parent/'training_complete.json') == cfg['parent_training_receipt_sha256']
    manifest = json.loads((parent/'input_manifest.json').read_text())
    assert manifest['input_role'] == 'head_fitting_sources_only' and not manifest['held_rows_transferred']
    for relative, sha in manifest['code_bindings'].items():
        assert digest(parent/'code'/relative) == sha
    trained = json.loads((parent/'training_complete.json').read_text())
    refs = {(r['group'], r['arm']): r for r in trained['artifacts']}
    assert len(refs) == 216 and len(manifest['groups']) == 108
    (home/'groups').mkdir(exist_ok=True)
    receipt_path = home/('replay_receipt.json' if a.replay else 'receipt.json')
    if receipt_path.exists():
        raise ValueError('Complete receipt exists; verify rather than rerun')
    phase = 'replay' if a.replay else 'diagnose'

    def beat(**kw):
        row = dict(phase=phase, pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'],
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
        write_path = home/'heartbeat.json'; write_path.write_text(json.dumps(row)+'\n')
        with (home/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    with (home/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); artifacts = []
        for entry in manifest['groups']:
            dest = home/'groups'/(entry['group']+'.json')
            if dest.exists() and a.resume and not a.replay:
                old = json.loads(dest.read_text())
                assert old['packet_sha256'] == entry['sha256']
                assert old['group'] == entry['group'] and old['parameter_updates'] == 0
            else:
                if dest.exists() and not a.replay:
                    raise ValueError('Existing group requires resume or replay')
                write_once(dest, group_readout(parent, entry, refs, cfg))
            artifacts.append(dict(path=str(dest.relative_to(home)), sha256=digest(dest)))
            beat(state='group_replayed' if a.replay else 'group_complete', groups=len(artifacts))
        write_once(receipt_path, dict(groups=108, heads=216, states=2, batches_per_state=4,
            gradient_batches=1728, artifacts=artifacts, phase=phase, replay_exact=a.replay,
            seconds=time.monotonic()-start, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=digest(home/'registration.json'), config_sha256=digest(home/'config.json'),
            job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), torch=torch.__version__, numpy=np.__version__,
            parameter_updates=0, held_outcomes_used=False, independent_roles_read=False))
        beat(state='complete', groups=108)


if __name__ == '__main__':
    main()
