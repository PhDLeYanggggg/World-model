"""Allocated-node evaluation of hash-bound, frozen selected-pool packets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.world_model import m3w_selected_pool_accounting as api


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value); h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def analyze(y, p, q, env, raw, action, rec, meta, supported):
    stats = api.account(y, p, q, env, raw, action, audit=True)
    assert not (action & ~supported).any()
    subset = api.account(y, p, q, env, raw & supported, action)
    records = []
    for recording in np.unique(rec):
        ix = rec == recording
        s = api.account(y[ix], p[ix], q[ix], env[ix], raw[ix], action[ix])
        records.append(dict(recording=str(recording), raw_rows=s['raw']['rows'],
            calibration_supported=bool(supported[ix].all()), kept_rows=s['kept']['rows'],
            unknown_kept=s['kept']['unknown'], raw_easy_risk=s['raw']['easy']['observed_risk'],
            kept_easy_risk=s['kept']['easy']['observed_risk'],
            easy_signed_bias_shift=s['easy_contrast']['signed_bias_shift'],
            new_known_violation=s['easy_contrast']['new_known_violation']))
    return dict(**meta, statistics=stats, recordings=records,
        unsupported_raw_rows=int((raw & ~supported).sum()),
        supported_pool_bias={k:dict(raw=subset['raw'][k]['signed_bias'], kept=subset['kept'][k]['signed_bias'],
            shift=subset[k+'_contrast']['signed_bias_shift']) for k in ('all','easy')},
        action_hash=array_hash(action), raw_action_hash=array_hash(raw),
        target_hash=array_hash(y), adjusted_prediction_hash=array_hash(q))


def compute(z, meta):
    rows = []
    assert array_hash(z['p']) == meta['identity']['parent_frozen_action']['raw_prediction_hash']
    for mode in ('harm', 'reference', 'joint'):
        q, action = z[mode+'_q'], z[mode+'_action']
        fr = meta['identity']['parent_frozen_action']
        assert array_hash(q) == fr['adjusted_hashes'][mode]
        assert array_hash(action) == fr['action_hashes'][mode]
        assert array_hash(z['raw']) == fr['action_hashes']['raw']
        rows.append(analyze(z['y'], z['p'], q, z['env'], z['raw'], action,
                            z['recordings'], meta['rows'][mode], z['calibration_support']))
    return dict(identity=meta['identity'], rows=rows, parent_risk_checks=12)


def once(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--home', required=True); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Allocated compute node required')
    home = Path(a.home); reg = json.loads((home/'recovery_registration.json').read_text())
    for name, h in reg['remote_code'].items(): assert digest(ROOT/name) == h
    manifest = json.loads((home/'input_manifest.json').read_text()); assert len(manifest['packets']) == 216
    assert not (home/'complete.json').exists(), 'Existing complete receipt: inspect, do not rerun'
    started = time.monotonic(); initial = []; parity = 0; risk_checks = 0
    with (home/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in ('compute', 'replay'):
            refs = []
            for entry in manifest['packets']:
                path = home/'inputs'/(entry['group']+'.npz'); assert digest(path) == entry['sha256']
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z['meta_json'])); result = compute(z, meta)
                if 'expected_local' in meta:
                    assert result == meta['expected_local'], 'Exact local/CREATE aggregate parity failed'
                    if phase == 'compute': parity += 1
                for row in result['rows']:
                    mode = row['mode']
                    for part, policy in (('raw','raw'), ('kept',mode)):
                        for prefix, field in (('all','selected_positive_harm_ratio'),('easy','selected_easy_positive_harm_ratio')):
                            observed=row['statistics'][part][prefix]['observed_risk']; expected=meta['parent_metrics'][policy][field]
                            assert (observed is None) == (expected is None)
                            if observed is not None: np.testing.assert_allclose(observed,expected,rtol=1e-10,atol=1e-10)
                            if phase == 'compute': risk_checks += 1
                out = home/'groups'/(entry['group']+'.json'); once(out, result)
                refs.append(dict(group=entry['group'], sha256=digest(out), bytes=out.stat().st_size))
                beat = dict(job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), phase=phase, groups=len(refs),
                            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                (home/'heartbeat.json').write_text(json.dumps(beat)+'\n')
                if len(refs)%18==0: print(json.dumps(beat),flush=True)
            if phase == 'compute': initial = refs
            else: assert initial == refs
        assert risk_checks == 2592 and parity == manifest['local_parity_groups']
        once(home/'complete.json', dict(groups=initial, transfer_views=216, exact_replay=True,
            local_parity_groups=parity, parent_risk_checks=risk_checks, seconds=time.monotonic()-started,
            job_id=os.environ['SLURM_JOB_ID'], new_parameter_updates=0,
            result_source='fresh_run_CREATE_frozen_diagnostic', independent_confirmation=False, policy_changed=False))
    print(json.dumps(dict(completed=True,views=216,local_parity=parity)),flush=True)


if __name__ == '__main__': main()
