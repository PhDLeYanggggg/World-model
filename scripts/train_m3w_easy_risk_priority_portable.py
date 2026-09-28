"""Registered paired repair on existing fitting-only CREATE packets."""
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
from src.world_model import m3w_easy_risk_priority as repair
from scripts.train_m3w_easy_hurdle_portable import digest, unpack
from scripts.run_m3w_easy_gradient_diagnostic import write_once

torch, api = repair.torch, repair.api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True); p.add_argument('--parent-home', required=True)
    p.add_argument('--phase', choices=['pilot', 'train', 'replay'], required=True)
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute node required')
    home, parent = Path(a.home), Path(a.parent_home)
    cfg = json.loads((home/'config.json').read_text()); reg = json.loads((home/'registration.json').read_text())
    assert cfg['arms'] == list(repair.ARMS) and cfg['groups'] == 108 and cfg['new_heads'] == 216
    assert cfg['auxiliary_norm_cap'] == repair.CAP and cfg['risk_budget'] == api.BUDGET
    assert not any(cfg[k] for k in ('independent_roles_read', 'threshold_search', 'deployment_changed',
                                    'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(cfg['interop_threads'])
    for rel, sha in reg['code_bindings'].items():
        assert digest(ROOT/rel) == sha
    assert digest(parent/'input_manifest.json') == cfg['parent_manifest_sha256']
    assert digest(parent/'training_complete.json') == cfg['parent_training_receipt_sha256']
    manifest = json.loads((parent/'input_manifest.json').read_text())
    assert manifest['input_role'] == 'head_fitting_sources_only' and not manifest['held_rows_transferred']
    previous = json.loads((parent/'training_complete.json').read_text())
    old_refs = {(r['group'], r['arm']): r for r in previous['artifacts']}
    assert len(manifest['groups']) == 108 and len(old_refs) == 216
    if a.phase == 'train':
        pilot = json.loads((home/'pilot.json').read_text())
        assert pilot['groups'] == 1 and pilot['updates_per_head'] == cfg['pilot_updates']
        assert pilot['registration_sha256'] == digest(home/'registration.json')
    if a.phase == 'replay':
        assert json.loads((home/'training_complete.json').read_text())['groups'] == 108
    receipt = home/({'pilot': 'pilot.json', 'train': 'training_complete.json', 'replay': 'replay.json'}[a.phase])
    if receipt.exists():
        raise ValueError('Completed receipt exists; verify, do not repeat')

    def beat(state, **kw):
        row = dict(state=state, phase=a.phase, pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'],
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
        (home/'heartbeat.json').write_text(json.dumps(row)+'\n')
        with (home/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    with (home/'train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); artifacts = []; fitting_summary = []; control_reproductions = 0
        for entry in manifest['groups']:
            name = entry['group']; packet = parent/'inputs'/(name+'.npz')
            assert digest(packet) == entry['sha256']
            arrays, pr, source_identity = unpack(packet)
            roles = source_identity['roles']
            for key in ('producer_sites', 'controller_sites', 'held_sites'):
                assert not set(arrays['sites']) & set(roles[key])
            identity = dict(source_identity=source_identity, registration_sha256=digest(home/'registration.json'))
            states = {}
            for arm in repair.ARMS:
                dest = home/('replay_heads' if a.phase == 'replay' else 'heads')/name/arm/'checkpoint.pt.gz'
                states[arm] = repair.fit(arrays['x'], arrays['u'], arrays['y'], arrays['sites'], arrays['recordings'],
                    arrays['frames'], arrays['env'], pr, arm=arm, seed=source_identity['seed'],
                    settings=cfg['head_training'], identity=identity, path=dest,
                    heartbeat=lambda **kw: beat(group=name, arm=arm, **kw),
                    resume=a.resume or (a.phase == 'train' and dest.exists()),
                    stop_at=cfg['pilot_updates'] if a.phase == 'pilot' else None)
                if a.phase == 'replay':
                    original = api.head.read_checkpoint(home/'heads'/name/arm/'checkpoint.pt.gz')
                    for key in original:
                        if key != 'seconds':
                            api.sampling.exact(original[key], states[arm][key])
                artifacts.append(dict(group=name, arm=arm, path=str(dest.relative_to(home)), sha256=digest(dest)))
                s = states[arm]
                fitting_summary.append(dict(group=name, arm=arm, step=s['step'],
                    fitting_seconds=s['seconds'],
                    first_monitor=s['trace'][0]['monitor'], last_monitor=s['trace'][-1]['monitor'],
                    cap_updates=s.get('cap_updates'), mean_alpha=s.get('alpha_sum', 0)/s['step'] if arm == 'risk_priority' else 1.,
                    min_risk_projection=s.get('min_risk_projection'), sample_hash=s['sample_hash'],
                    query_draws=s['query_draws'], row_draws=s['row_draws'], unknown_rows_sampled=s['unknown_rows_sampled']))
            repair.assert_matched(states['uncapped'], states['risk_priority'])
            if a.phase != 'pilot':
                ref = old_refs[name, 'supervised']; checkpoint = parent/ref['path']
                assert digest(checkpoint) == ref['sha256']
                repair.assert_parent_control(api.head.read_checkpoint(checkpoint), states['uncapped'])
                control_reproductions += 1
            beat('paired_fit_complete', group=name, groups=len(artifacts)//2)
            if a.phase != 'train':
                break
        n = len(artifacts)//2
        assert n == (108 if a.phase == 'train' else 1)
        write_once(receipt, dict(groups=n, heads=2*n, artifacts=artifacts, fitting_summary=fitting_summary, phase=a.phase,
            updates_per_head=cfg['pilot_updates'] if a.phase == 'pilot' else cfg['head_training']['steps'],
            observed_fitting_seconds=sum(s['fitting_seconds'] for s in fitting_summary),
            control_parent_states_exact=control_reproductions, repair_first_pair_replay_exact=a.phase == 'replay',
            seconds=time.monotonic()-start, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), torch=torch.__version__,
            numpy=repair.np.__version__, registration_sha256=digest(home/'registration.json'),
            config_sha256=digest(home/'config.json'), held_outcomes_used=False, independent_roles_read=False))
        beat('complete', groups=n)


if __name__ == '__main__':
    main()
