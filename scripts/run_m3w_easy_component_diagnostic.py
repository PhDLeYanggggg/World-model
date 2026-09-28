"""Fitting-only component accounting using immutable CREATE packets and heads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_m3w_easy_hurdle_portable import api, digest, unpack, np, torch
from scripts.run_m3w_easy_gradient_diagnostic import write_once
from src.world_model.m3w_easy_component_diagnostic import compose, diagnose
from src.world_model.m3w_easy_risk_priority import assert_matched


def read_group(parent, trained, entry, refs):
    packet = parent/'inputs'/(entry['group']+'.npz')
    assert digest(packet) == entry['sha256']
    a, pr, identity = unpack(packet); roles = identity['roles']
    assert set(a['sites']) == set(roles['training_sites'])
    for key in ('producer_sites', 'controller_sites', 'held_sites'):
        assert not set(a['sites']) & set(roles[key])
    assert np.array_equal(np.isfinite(a['y']).all(1), pr['known'])
    pred, states = {}, {}
    for arm in ('uncapped', 'risk_priority'):
        ref = refs[entry['group'], arm]; checkpoint = trained/ref['path']
        assert digest(checkpoint) == ref['sha256']
        state = api.head.read_checkpoint(checkpoint); states[arm] = state
        assert state['identity']['source_identity'] == identity and state['step'] == 2000
        assert state['unknown_rows_sampled'] == 0
        norm = api.preprocess(a['x'], a['u'], a['env'], pr)[2]
        api.sampling.exact(state['norm'], norm)
        values = api.predict(state, a['x'], a['u'], a['env'])
        np.testing.assert_allclose(values[:, 3], compose(values[:, :3]), rtol=1e-5, atol=1e-7)
        pred[arm] = values[:, :3]
        assert digest(checkpoint) == ref['sha256']
    assert_matched(states['uncapped'], states['risk_priority'])
    report = diagnose(pred['uncapped'], pred['risk_priority'], a['y'], a['sites'], a['recordings'], a['frames'])
    return dict(group=entry['group'], identity=identity, packet_sha256=entry['sha256'],
        checkpoints={arm: refs[entry['group'], arm]['sha256'] for arm in pred},
        held_outcomes_used=False, independent_roles_read=False, **report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True); p.add_argument('--parent-home', required=True)
    p.add_argument('--trained-home', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--replay', action='store_true'); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute node required')
    home, parent, trained = map(Path, (a.home, a.parent_home, a.trained_home))
    cfg = json.loads((home/'config.json').read_text()); reg = json.loads((home/'registration.json').read_text())
    assert cfg['groups'] == 108 and cfg['parameter_updates'] == 0 and cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('held_outcomes_used', 'independent_roles_read', 'threshold_search',
        'deployment_changed', 'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(cfg['interop_threads'])
    for rel, sha in reg['code_bindings'].items():
        assert digest(ROOT/rel) == sha
    assert digest(parent/'input_manifest.json') == cfg['parent_manifest_sha256']
    assert digest(trained/'training_complete.json') == cfg['trained_receipt_sha256']
    manifest = json.loads((parent/'input_manifest.json').read_text())
    training = json.loads((trained/'training_complete.json').read_text())
    assert manifest['input_role'] == 'head_fitting_sources_only' and not manifest['held_rows_transferred']
    refs = {(r['group'], r['arm']): r for r in training['artifacts']}
    assert len(refs) == 216 and len(manifest['groups']) == 108
    for rel, sha in manifest['code_bindings'].items():
        assert digest(parent/'code'/rel) == sha
    phase = 'replay' if a.replay else 'diagnose'
    receipt = home/('replay_receipt.json' if a.replay else 'receipt.json')
    if receipt.exists():
        raise ValueError('Complete receipt exists; do not overwrite')
    (home/'groups').mkdir(exist_ok=True)

    def beat(**kw):
        row = dict(phase=phase, pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'],
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
        (home/'heartbeat.json').write_text(json.dumps(row)+'\n')
        with (home/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    with (home/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); artifacts = []
        for entry in manifest['groups']:
            dest = home/'groups'/(entry['group']+'.json'); before = time.monotonic()
            if dest.exists() and a.resume and not a.replay:
                old = json.loads(dest.read_text())
                assert old['packet_sha256'] == entry['sha256'] and old['parameter_updates'] == 0
                assert old['registration_sha256'] == digest(home/'registration.json')
                assert old['checkpoints'] == {arm: refs[entry['group'], arm]['sha256'] for arm in cfg['arms']}
                assert not old['held_outcomes_used'] and not old['policy_actions_computed']
            else:
                if dest.exists() and not a.replay: raise ValueError('Existing group requires resume or replay')
                row = read_group(parent, trained, entry, refs)
                row['registration_sha256'] = digest(home/'registration.json')
                write_once(dest, row)
            artifacts.append(dict(path=str(dest.relative_to(home)), sha256=digest(dest)))
            beat(state='group_replayed' if a.replay else 'group_complete', groups=len(artifacts),
                 group_seconds=time.monotonic()-before, projected_seconds=(time.monotonic()-start)*108/len(artifacts))
        write_once(receipt, dict(groups=108, heads=216, artifacts=artifacts, phase=phase,
            replay_exact=a.replay, seconds=time.monotonic()-start, job_id=os.environ['SLURM_JOB_ID'],
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=digest(home/'registration.json'), config_sha256=digest(home/'config.json'),
            parameter_updates=0, held_outcomes_used=False, independent_roles_read=False,
            policy_actions_computed=False, torch=torch.__version__, numpy=np.__version__))
        beat(state='complete', groups=108)


if __name__ == '__main__': main()
