"""Allocated-node checkpoint audit and first-group exact full fitting replay."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_m3w_easy_hurdle_portable import unpack, digest, api, np, torch


def check_state(s, identity, cfg, arm):
    assert s['identity']==identity and s['arm']==arm
    assert s['settings']==cfg['head_training'] and s['step']==cfg['head_training']['steps']
    assert s['unknown_rows_sampled']==0 and s['query_draws']==s['step']*cfg['head_training']['query_batch_size']
    assert set(s['norm']['training_sites'])==set(identity['roles']['training_sites'])
    assert s['probability_calibration_certificate'] is False
    assert all(torch.isfinite(v).all() for v in s['model'].values())
    assert all(np.isfinite(list(row['monitor'].values())).all() for row in s['trace'])
    assert s['trace'][0]['step']==0 and s['trace'][-1]['step']==s['step']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--home',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated compute node required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    home=Path(a.home);cfg=json.loads((home/'config.json').read_text());manifest=json.loads((home/'input_manifest.json').read_text())
    trained=json.loads((home/'training_complete.json').read_text())
    assert trained['complete'] and trained['groups']==108 and not trained['held_outcomes_used']
    assert trained['manifest_sha256']==digest(home/'input_manifest.json')
    for relative,sha in manifest['code_bindings'].items():assert digest(ROOT/relative)==sha
    output=home/'training_audit.json'
    if output.exists():raise ValueError('Completed verification exists; retrieve and verify rather than overwrite')
    refs={(r['group'],r['arm']):r for r in trained['artifacts']};assert len(refs)==216

    def beat(state,**kw):
        row=dict(state=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],**kw)
        (home/'verification_heartbeat.json').write_text(json.dumps(row)+'\n')
        with (home/'verification_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    with (home/'verification.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();rows=[];first=None
        for group in manifest['groups']:
            path=home/'inputs'/(group['group']+'.npz');assert digest(path)==group['sha256']
            with np.load(path,allow_pickle=False) as z:identity=json.loads(str(z['identity_json']))
            assert identity['registration']['sha256']==json.loads((home/'.owner.json').read_text())['registration_sha256']
            states={}
            for arm in api.ARMS:
                ref=refs[(group['group'],arm)];path=home/ref['path'];assert digest(path)==ref['sha256']
                s=api.head.read_checkpoint(path);check_state(s,identity,cfg,arm);states[arm]=s
                rows.append(dict(group=group['group'],arm=arm,step=s['step'],seconds=s['seconds'],
                    first=s['trace'][0],last=s['trace'][-1],sample_hash=s['sample_hash'],
                    query_draws=s['query_draws'],row_draws=s['row_draws'],unknown_rows_sampled=0,
                    checkpoint_sha256=ref['sha256']))
            api.assert_matched(states['marginal'],states['supervised'])
            if first is None:first=(group,states)
            beat('checkpoint_pair_verified',groups=len(rows)//2)
        group,original=first;arrays,pr,identity=unpack(home/'inputs'/(group['group']+'.npz'))
        for arm in api.ARMS:
            path=home/'fit_replay'/group['group']/arm/'checkpoint.pt.gz'
            state=api.fit(*[arrays[k] for k in ('x','u','y','sites','recordings','frames','env')],pr,
                arm=arm,seed=identity['seed'],settings=cfg['head_training'],identity=identity,path=path,
                heartbeat=lambda **kw:beat(group=group['group'],arm=arm,**kw),resume=a.resume)
            for key in state:
                if key!='seconds':api.sampling.exact(original[arm][key],state[key])
        result=dict(groups=108,heads=216,model_updates=sum(r['step'] for r in rows),paired_initialization_and_sample_chain_exact=True,
            all_checkpoints_finite_and_hash_verified=True,first_group_full_training_replay_exact_except_elapsed=True,
            replay_heads=2,replay_updates=4000,full_216head_retraining_replay=False,rows=rows,
            manifest_sha256=digest(home/'input_manifest.json'),training_receipt_sha256=digest(home/'training_complete.json'),
            verifier_sha256=digest(__file__),job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),seconds=time.monotonic()-start,
            held_outcomes_used=False,independent_roles_read=False,scientific_efficacy_evaluated=False)
        with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        beat('verification_complete',groups=108,heads=216)


if __name__=='__main__':main()
