"""Register, stream frozen diagnostics, submit once and collect small evidence."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_inner_separability as parent
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote, HANDOFF
from scripts.export_m3w_easy_hurdle_create import closure, remote
from scripts.m3w_packet_stream import PacketStream

NAME = 'european_boundary_diagnostic_v1'
PUBLIC = parent.PUBLIC.parent/NAME; PRIVATE = parent.PRIVATE.parent/NAME
REMOTE = '/users/k24101830/m3w/'+NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = parent.digest, parent.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); seal = parent.PUBLIC/'verification.json'
    assert digest(seal) == cfg['parent_seal_sha256']
    v = json.loads(seal.read_text())
    for k, h in v['source_bindings'].items(): assert digest(ROOT/k) == h
    for k, h in v['artifacts'].items(): assert digest(parent.PUBLIC/k) == h
    code = closure(ROOT, ['scripts.run_m3w_boundary_diagnostic'])
    control = [Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_boundary_diagnostic.py',
               ROOT/'scripts/m3w_packet_stream.py', ROOT/'scripts/manage_m3w_easy_gradient_diagnostic.py']
    return cfg, dict(parent_seal_sha256=digest(seal),
        remote_code={str(p.relative_to(ROOT)): digest(p) for p in code},
        controls={str(p.relative_to(ROOT)): digest(p) for p in control},
        groups=288, independent_roles_read=False, parameter_updates=0)


RECEIVER = r'''
import hashlib,json,os,pathlib,re,sys
root=pathlib.Path(sys.argv[1]); experiment=sys.argv[2]
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name==experiment
assert json.loads((root/'.owner.json').read_text())['experiment']==experiment
print(json.dumps({'ready':True}),flush=True)
stream=sys.stdin.buffer
while True:
    line=stream.readline(4097)
    if not line:break
    assert len(line)<=4096 and line.endswith(b'\n')
    h=json.loads(line);name=h['group'];size=h['bytes']
    assert re.fullmatch(r'[A-Za-z0-9_-]+',name) and 0<size<=134217728
    chunks=[];remaining=size
    while remaining:
        chunk=stream.read(min(remaining,2**20));assert chunk
        chunks.append(chunk);remaining-=len(chunk)
    raw=b''.join(chunks);assert hashlib.sha256(raw).hexdigest()==h['sha256']
    path=root/'inputs'/(name+'.npz');path.parent.mkdir(exist_ok=True)
    if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==h['sha256']
    else:
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    print(json.dumps(dict(group=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())),flush=True)
'''


class BoundaryStream(PacketStream):
    def __init__(self, ssh):
        self.command = ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,REMOTE,NAME])]
        self.process = None


def heartbeat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.base.inter.json_write(PRIVATE/'heartbeat.json', row)
    print(json.dumps(row), flush=True)


def export(cfg, reg, resume=False):
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    setup = r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name==p['name']
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
root.mkdir(exist_ok=True)
def once(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==value
    else:path.write_text(value)
once(root/'.owner.json',json.dumps({'experiment':p['name'],'project':'M3W'})+'\n')
for name in ('config','registration'):
    once(root/(name+'.json'),json.dumps(p[name],indent=2,allow_nan=False)+'\n')
for name,text in p['code'].items():
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['remote_code'][name]
    assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    once(root/'code'/name,text)
refs=[]
for path in sorted((root/'inputs').glob('*.npz')):
    refs.append(dict(group=path.stem,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps({'existing':refs}))
'''
    existing = remote(ssh, setup, payload=json.dumps(dict(root=REMOTE,name=NAME,config=cfg,registration=reg,
        code={k:(ROOT/k).read_text() for k in reg['remote_code']})).encode())['existing']
    if existing and not resume: raise ValueError('Use resume after verifying remote packets')
    cached = {r['group']:r for r in existing}; refs=[]; proofs=[]; start=time.monotonic()
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1)
    _, _, data, jobs, oid, _, _, _ = parent.old.load()
    causal = {k:data[k] for k in parent.old.parent.CAUSAL_KEYS}
    fits = {Path(r['path']).parent.name:r for r in json.loads((parent.PUBLIC/'training_freeze.json').read_text())['groups']}
    with BoundaryStream(ssh) as stream:
        for c in parent.base.floor_api.contexts(causal,jobs,oid):
            tails = {}; tail_counts = {}
            for site in parent.sources(c):
                at, ids, x, env, y, pr, ident = parent.training_arrays(c,data,site)
                tails[site] = {event:float(np.quantile(y[np.isfinite(y[:,col]) & (y[:,col]>0),col],.95))
                    if (np.isfinite(y[:,col]) & (y[:,col]>0)).any() else 0. for event,col in [('all',1),('easy',4)]}
                tail_counts[site] = {event:int((np.isfinite(y[:,col]) & (y[:,col]>0)).sum())
                                     for event,col in [('all',1),('easy',4)]}
            directions = [('fitting',None,s) for s in parent.sources(c)]
            directions += [('internal_transfer',pair,s) for pair,(sites,_) in enumerate(c['pairs']) for s in sites]
            for phase,pair,site in directions:
                if phase == 'internal_transfer':
                    at, values, masks, meta, pr = parent.view_predictions(c,causal,pair,site,fits)
                    name=c['name']+f'_pair{pair}_from_'+site
                    original=json.loads((parent.PRIVATE/'decisions'/(name+'.json')).read_text())
                    assert meta['score_hashes']==original['score_hashes']
                    with np.load(ROOT/original['arrays']['path'],allow_pickle=False) as z:
                        for k,value in masks.items():np.testing.assert_array_equal(z[k],value)
                    source=meta['eval_site']; outer=meta['outer_held_sites']
                else:
                    at, ids, x, env, y, pr, ident = parent.training_arrays(c,data,site)
                    doc=parent.checked(fits[c['name']+'_fit_'+site]);assert doc['identity']==ident
                    values={}; support=None
                    for arm,ref in doc['artifacts'].items():
                        assert parent.base.artifact(ROOT/ref['path'])==ref
                        state=parent.api.read_checkpoint(ROOT/ref['path'])
                        values[arm],ok=parent.api.predict(state,x,env)
                        if support is not None:np.testing.assert_array_equal(support,ok)
                        support=ok
                    masks=parent.api.decisions(values,c['moving'][at],support,data['recordings'][ids],data['frames'][ids],ids)
                    masks.update(ids=ids,support=support)
                    name=c['name']+'_fitting_'+site;source=site;outer=[]
                ids=masks['ids']
                cv,_,(floor,_),(neural,_)=parent.base.floor_api.costs(c,data,at)
                y=parent.api.targets(cv,floor,neural,c['job']['design']['easy_cut'])
                _,recording=np.unique(data['recordings'][ids],return_inverse=True)
                arrays=dict(ids=ids,targets=y,envelope=c['env'][at],moving=c['moving'][at],support=masks['support'],
                    recording=recording,frame=data['frames'][ids])
                for arm in cfg['arms']:
                    arrays.update({arm+'_pred':values[arm],arm+'_selected':masks[arm],arm+'_matched':masks[arm+'_matched']})
                meta=dict(view=name,phase=phase,site=source,train_site=site,outer_held_sites=outer,
                    source_roles=dict(producer_sites=c['producer_sites'],controller_sites=c['controller_sites']),
                    cost_scale=pr['scale'],tail_cuts=tails[site],seed=c['job']['old_identity']['seed'],
                    tail_positive_training_counts=tail_counts[site],
                    causal_score_sha256={arm:parent.base.inter.array_hash(values[arm]) for arm in cfg['arms']},
                    targets_used_only_for_diagnostic=True)
                arrays['meta_json']=np.array(json.dumps(meta,sort_keys=True))
                buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
                expected=dict(group=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                assert len(raw)<=cfg['max_remote_packet_bytes']
                if name in cached:assert cached[name]==expected
                else:assert stream.send(name,raw)==expected
                refs.append(expected)
                if len(proofs)<2 or (phase=='internal_transfer' and len(proofs)==2):
                    from src.world_model.m3w_boundary_diagnostic import diagnostic
                    proof=diagnostic({k:v for k,v in arrays.items() if k!='meta_json'},meta,cfg)
                    proofs.append(dict(group=name,result=proof))
                heartbeat(state='packet_verified',groups=len(refs),group=name,bytes=sum(r['bytes'] for r in refs))
    assert len(refs)==288
    manifest=dict(packets=refs,registration_sha256=digest(PUBLIC/'registration.json'),
        parent_seal_sha256=cfg['parent_seal_sha256'],independent_roles_read=False)
    code="import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/'manifest.json'; raw=sys.stdin.read(); assert not p.exists() or p.read_text()==raw; p.write_text(raw) if not p.exists() else None; print(json.dumps({'complete':True}))"
    assert remote(ssh,code,[REMOTE],(json.dumps(manifest,indent=2)+'\n').encode())['complete']
    immutable(PRIVATE/'local_parity_proofs.json',proofs)
    immutable(PUBLIC/'packet_manifest.json',manifest)
    immutable(PUBLIC/'transfer_receipt.json',dict(groups=288,bytes=sum(r['bytes'] for r in refs),
        seconds=time.monotonic()-start,local_large_cache_bytes=0,cached_verified_packets=len(cached),
        new_packets=288-len(cached),destination='isolated_authorized_M3W_CREATE',independent_roles_read=False))


def submit(reg):
    parent.base.inter.committed(PUBLIC/'packet_manifest.json')
    assert not (PRIVATE/'submission.json').exists(), 'Inspect prior/uncertain submission'
    script=f'''#!/bin/bash -l
#SBATCH --job-name=m3w_boundary_diag
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=run-%j.out
#SBATCH --error=run-%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/run_m3w_boundary_diagnostic.py --home "$PWD" --resume
'''
    code=r'''
import json,pathlib,subprocess,sys,hashlib
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root.parent==pathlib.Path('/users/k24101830/m3w')
assert json.loads((root/'.owner.json').read_text())['experiment']==p['name']
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==p['manifest_sha256']
assert not (root/'submission_intent.json').exists() and not (root/'submission_receipt.json').exists()
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and 'm3w_boundary_diag' not in q.stdout
(root/'run.sh').write_text(p['script'])
(root/'submission_intent.json').write_text(json.dumps({'script_sha256':hashlib.sha256(p['script'].encode()).hexdigest()})+'\n')
r=subprocess.run(['sbatch','--parsable','run.sh'],cwd=root,capture_output=True,text=True,timeout=30)
out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:out['job_id']=r.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
(root/'submission_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
    response=call_remote(code,dict(root=REMOTE,name=NAME,script=script,
        manifest_sha256=digest(PUBLIC/'packet_manifest.json')))
    immutable(PRIVATE/'submission.json',response)
    print(json.dumps(dict(returncode=response['returncode'],response=response.get('stdout'),uncertain=response['returncode'] is None)))


def inspect(collect=False):
    code=r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root']);out={}
assert json.loads((root/'.owner.json').read_text())['experiment']==p['name']
for name in ('submission_receipt','heartbeat','receipt'):
    path=root/(name+'.json')
    if path.exists():out[name]=json.loads(path.read_text())
job=out.get('submission_receipt',{}).get('job_id')
if job:
    q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=20)
    out['accounting']=dict(code=q.returncode,stdout=q.stdout)
if p['collect']:
    assert out['accounting']['code']==0 and out['accounting']['stdout'].startswith('COMPLETED|0:0|')
    receipt=out['receipt'];assert receipt['groups']==288 and receipt['full_replay_exact']
    for r in receipt['artifacts']:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
    assert hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()==receipt['summary_sha256']
    out['summary']=json.loads((root/'summary.json').read_text())
    out['proofs']=[dict(group=name,result=json.loads((root/'groups'/(name+'.json')).read_text())) for name in p['proof_names']]
else:
    if 'receipt' in out:out['receipt']={k:v for k,v in out['receipt'].items() if k!='artifacts'}
print(json.dumps(out))
'''
    proofs=json.loads((PRIVATE/'local_parity_proofs.json').read_text()) if collect else []
    response=call_remote(code,dict(root=REMOTE,name=NAME,collect=collect,proof_names=[p['group'] for p in proofs]))
    if response['returncode']!=0:
        print(json.dumps(dict(returncode=response['returncode'],unavailable_not_terminal=True,error=response.get('stderr','')[-1500:])))
        return
    out=json.loads(response['stdout'])
    if collect:
        assert len(response['stdout'].encode())<8*2**20
        assert out['receipt']['registration_sha256']==digest(PUBLIC/'registration.json')
        assert out['receipt']['manifest_sha256']==digest(PUBLIC/'packet_manifest.json')
        cfg=json.loads(CONFIG.read_text())
        assert out['receipt']['config_sha256']==hashlib.sha256((json.dumps(cfg,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        for local,other in zip(proofs,out.pop('proofs')):
            remote_result=other['result'];remote_result.pop('packet_sha256')
            assert local['group']==other['group']
            # Linux/macOS reductions can differ at roundoff only; never compare loosely by a percent.
            compare_tree(local['result'],remote_result)
        existing=json.loads((parent.PUBLIC/'readout.json').read_text())
        metrics={(r['view'],r['policy']):r['metric'] for r in existing['rows']}
        checks=0
        for row in out['summary']['parent_readout_comparison']:
            for arm,scopes in row['risk'].items():
                for scope,events in scopes.items():
                    policy=arm+('_matched' if scope=='matched' else '')
                    for event,risk in events.items():
                        key='selected_positive_harm_ratio' if event=='all' else 'selected_easy_positive_harm_ratio'
                        compare_tree(risk,metrics[row['view'],policy][key]);checks+=1
        out['local_parent_risk_checks']=checks;out['local_packet_parity_checks']=len(proofs)
        immutable(PRIVATE/'collected.json',out)
        immutable(PUBLIC/'compute_receipt.json',out['receipt'])
        immutable(PUBLIC/'summary.json',out['summary'])
        print(json.dumps(dict(collected=True,groups=288,parent_risk_checks=checks,local_parity_packets=len(proofs))))
    else:print(json.dumps(out,indent=2))


def compare_tree(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare_tree(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_tree(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-9)
    else:assert a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','export','submit','inspect','collect']);p.add_argument('--resume',action='store_true')
    a=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if a.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered288 frozen diagnostic groups');return
    assert reg==json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if a.phase=='export':export(cfg,reg,a.resume)
    elif a.phase=='submit':submit(reg)
    else:inspect(a.phase=='collect')


if __name__=='__main__':main()
