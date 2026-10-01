"""Resource-only, memory-streamed continuation of frozen selected-pool accounting."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_selected_pool_accounting as run
from scripts.manage_m3w_boundary_diagnostic import RECEIVER
from scripts.m3w_packet_stream import PacketStream
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote, HANDOFF

REMOTE='/users/k24101830/m3w/'+run.NAME
PUBLIC,PRIVATE=run.PUBLIC,run.PRIVATE


def registration():
    cfg, _=run.registration()
    code=['scripts/run_m3w_selected_pool_portable.py','src/world_model/m3w_selected_pool_accounting.py']
    controls=[Path(__file__),ROOT/'tests/test_m3w_selected_pool_portable.py',PUBLIC/'resource_recovery.md',
              ROOT/'scripts/m3w_packet_stream.py',ROOT/'scripts/manage_m3w_boundary_diagnostic.py',
              ROOT/'scripts/manage_m3w_easy_gradient_diagnostic.py']
    return dict(original_registration_sha256=run.digest(PUBLIC/'registration.json'),
        remote_code={p:run.digest(ROOT/p) for p in code},
        controls={str(p.relative_to(ROOT)):run.digest(p) for p in controls},
        input_byte_cap=512*2**20, groups=216, risk_budget=cfg['risk_budget'],
        source_groups_already_exact=72, new_parameter_updates=0, policy_changed=False,
        independent_roles_read=False, stage5c_executed=False,smc_enabled=False)


def remote(code,payload):
    r=call_remote(code,payload)
    if r['returncode']!=0: raise RuntimeError(json.dumps(r))
    return json.loads(r['stdout'])


def export(reg):
    setup=r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name==p['name']
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
root.mkdir(exist_ok=True)
def once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
once(root/'.owner.json',json.dumps({'experiment':p['name'],'project':'M3W'})+'\n')
once(root/'recovery_registration.json',json.dumps(p['registration'],indent=2)+'\n')
for name,text in p['files'].items():
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['remote_code'][name]
    assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    once(root/'code'/name,text)
print(json.dumps({'prepared':True,'existing_packets':len(list((root/'inputs').glob('*.npz')))}))
'''
    remote(setup,dict(root=REMOTE,name=run.NAME,registration=reg,
        files={k:(ROOT/k).read_text() for k in reg['remote_code']}))
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    stream=PacketStream.__new__(PacketStream)
    stream.command=ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,REMOTE,run.NAME])];stream.process=None
    run.core.torch.set_num_threads(4);run.core.torch.set_num_interop_threads(1)
    _,_,data,jobs,oid,_,_,_=run.inner.old.load(); fitted=run.parent.docs()
    frozen={r['view']:r for r in json.loads((run.parent.PUBLIC/'decision_freeze.json').read_text())['rows']}
    old={(r['view'],r['policy']):r['metric'] for r in json.loads((run.parent.PUBLIC/'readout.json').read_text())['rows']}
    refs=[];total=0;parity=0;started=time.monotonic()
    with stream:
        for c,at,ids,predictions,_,pr,oldmeta in run.parent.parent.views(data,jobs,oid):
            support=run.forest.api.causal_inputs(c['x'][at],c['env'][at],pr)[1]
            env,moving,rec=c['env'][at],c['moving'][at],data['recordings'][ids].astype(str)
            for seed in run.parent.parent.api.SEEDS:
                group=c['name']+'_fit_'+oldmeta['source'];view=oldmeta['view']+'_head'+str(seed)
                doc,fr=fitted[group,seed],frozen[view];p=predictions[seed]
                raw=run.parent.api.eligible(p,moving,support)
                assert run.base.inter.array_hash(raw)==fr['action_hashes']['raw']
                arrays=dict(p=p,env=env,raw=raw,recordings=rec,
                            calibration_support=np.full(len(ids),doc['final']['supported'],bool))
                metas={}
                for mode in run.parent.api.MODES:
                    q=run.parent.api.adjust(p,env,doc['final'],mode)
                    action=run.parent.api.eligible(q,moving,support)
                    assert run.base.inter.array_hash(q)==fr['adjusted_hashes'][mode]
                    assert run.base.inter.array_hash(action)==fr['action_hashes'][mode]
                    arrays[mode+'_q']=q;arrays[mode+'_action']=action
                    metas[mode]=dict(view=view,group=group,source=oldmeta['source'],site=oldmeta['site'],
                        head_seed=seed,mode=mode,role='transfer',source_screen=doc['source'][mode]['oof']['finite_completion_supported'])
                cv,_,(floor,_),(neural,_)=run.base.floor_api.costs(c,data,at)
                arrays['y']=run.core.targets(cv,floor,neural,c['job']['design']['easy_cut'])
                identity=dict(registration_sha256=run.digest(PUBLIC/'registration.json'),
                    parent_frozen_action=fr,ids_hash=run.base.inter.array_hash(ids))
                meta=dict(identity=identity,rows=metas,
                    parent_metrics={k:{f:old[view,k][f] for f in ('selected_positive_harm_ratio','selected_easy_positive_harm_ratio')}
                                    for k in ('raw','harm','reference','joint')})
                local=PRIVATE/'transfer'/(view+'.json')
                if local.exists():
                    meta['expected_local']=json.loads(local.read_text())
                    assert meta['expected_local']['identity']==identity;parity+=1
                arrays['meta_json']=np.array(json.dumps(meta,sort_keys=True))
                buf=io.BytesIO();np.savez_compressed(buf,**arrays);packet=buf.getvalue()
                total+=len(packet);assert total<=reg['input_byte_cap']
                refs.append(stream.send(view,packet))
                if len(refs)%18==0:run.beat(state='CREATE_input_stream',groups=len(refs),bytes=total)
    assert len(refs)==216
    manifest=dict(packets=refs,bytes=total,local_parity_groups=parity,
                  recovery_registration_sha256=run.digest(PUBLIC/'recovery_registration.json'))
    code=r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert json.loads((root/'.owner.json').read_text())['experiment']==p['name']
for r in p['manifest']['packets']:
    f=root/'inputs'/(r['group']+'.npz');assert f.stat().st_size==r['bytes']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
f=root/'input_manifest.json';text=json.dumps(p['manifest'],indent=2)+'\n'
if f.exists():assert f.read_text()==text
else:f.write_text(text)
print(json.dumps({'verified_packets':len(p['manifest']['packets'])}))
'''
    remote(code,dict(root=REMOTE,name=run.NAME,manifest=manifest))
    run.immutable(PUBLIC/'create_input_manifest.json',manifest)
    run.immutable(PUBLIC/'create_export_receipt.json',dict(groups=216,bytes=total,seconds=time.monotonic()-started,
        no_local_array_cache=True,remote_compute='not_run',local_parity_groups=parity))


def submit():
    path=PRIVATE/'create_submission.json';assert not path.exists(),'Inspect existing intent before retry'
    script='''#!/bin/bash -l
#SBATCH --job-name=m3w_selected_pool_v1
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostic-%j.out
#SBATCH --error=diagnostic-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/run_m3w_selected_pool_portable.py --home "$PWD"
'''
    code=r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root==pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
assert hashlib.sha256((root/'input_manifest.json').read_bytes()).hexdigest()==p['manifest_hash']
assert not (root/'submission_intent.json').exists() and not (root/'submission_receipt.json').exists()
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=30)
assert q.returncode==0 and 'm3w_selected_pool_v1' not in q.stdout
(root/'run.sh').write_text(p['script'])
(root/'submission_intent.json').write_text(json.dumps({'manifest':p['manifest_hash'],'queue':q.stdout}))
r=subprocess.run(['sbatch','--parsable','run.sh'],cwd=root,capture_output=True,text=True,timeout=30)
out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,jobs_submitted=int(r.returncode==0))
if r.returncode==0:
    job=r.stdout.strip().split(';')[0];assert job.isdigit();out['job_id']=job
(root/'submission_receipt.json').write_text(json.dumps(out))
print(json.dumps(out))
'''
    response=call_remote(code,dict(root=REMOTE,manifest_hash=run.digest(PUBLIC/'create_input_manifest.json'),script=script))
    run.immutable(path,response)
    if response['returncode']!=0:raise RuntimeError('Submission outcome requires inspection; do not duplicate')
    result=json.loads(response['stdout']);run.immutable(PUBLIC/'create_submission_receipt.json',result)
    assert result['returncode']==0;print(json.dumps(result))


def inspect():
    code=r'''
import json,pathlib,subprocess,sys
root=pathlib.Path(json.loads(sys.stdin.read())['root']);r=root/'submission_receipt.json'
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
out={'submission':json.loads(r.read_text()) if r.exists() else None}
if out['submission'] and 'job_id' in out['submission']:
    j=out['submission']['job_id']
    a=subprocess.run(['sacct','-j',j,'-X','--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=30)
    out['accounting']=dict(returncode=a.returncode,stdout=a.stdout,stderr=a.stderr)
    for suffix in ('out','err'):
        p=root/('diagnostic-'+j+'.'+suffix);out[suffix]=p.read_text()[-6000:] if p.exists() else None
for name in ('heartbeat.json','complete.json'):
    p=root/name;out[name]=json.loads(p.read_text()) if p.exists() else None
print(json.dumps(out))
'''
    result=remote(code,dict(root=REMOTE))
    path=PRIVATE/('create_observation_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json')
    run.immutable(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='complete.json'}))
    if result['complete.json']: print(json.dumps({'complete_receipt':True,'views':result['complete.json']['transfer_views']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['register','export','submit','inspect']);a=p.parse_args()
    reg=registration()
    if a.phase=='register':run.immutable(PUBLIC/'recovery_registration.json',reg);print('Recovery registered');return
    assert json.loads((PUBLIC/'recovery_registration.json').read_text())==reg
    run.base.inter.committed(PUBLIC/'recovery_registration.json')
    if a.phase=='export':export(reg)
    elif a.phase=='submit':submit()
    else:inspect()


if __name__=='__main__':main()
