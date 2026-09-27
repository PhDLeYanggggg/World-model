"""Stream fitting-only packets to isolated M3W storage without local large files."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import numpy as np
from scripts import run_m3w_easy_hurdle as run
from scripts.prepare_m3w_create_runtime import HANDOFF


def closure(root, modules):
    todo=list(modules); seen=set(); paths=set()
    while todo:
        module=todo.pop()
        if module in seen:
            continue
        seen.add(module); p=root/(module.replace('.','/')+'.py')
        if not p.is_file():
            continue
        paths.add(p)
        for parent in p.parents:
            if parent==root:
                break
            init=parent/'__init__.py'
            if init.is_file():
                todo.append(str(init.relative_to(root).with_suffix('')).replace('/','.'))
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.Import):
                todo.extend(a.name for a in node.names if a.name.startswith(('src.','scripts.')))
            elif isinstance(node,ast.ImportFrom) and node.module and (node.module in ('src','scripts') or node.module.startswith(('src.','scripts.'))):
                todo.append(node.module);todo.extend(node.module+'.'+a.name for a in node.names)
    return sorted(paths)


def remote(ssh, code, args=(), payload=b''):
    p=subprocess.run(ssh+[shlex.join(['/usr/bin/python3','-c',code,*args])],input=payload,capture_output=True,timeout=60)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
    return json.loads(p.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true')
    p.add_argument('--runtime-revision',type=int,choices=[1,2],default=2);a=p.parse_args()
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'StrictHostKeyChecking=yes' in ssh and 'BatchMode=yes' in ssh
    suffix='' if a.runtime_revision==1 else '_v2'
    submitted=json.loads((run.PRIVATE/('create_runtime_submission'+suffix+'.json')).read_text())
    runtime=json.loads(submitted['response']['stdout'])['trial_path']
    check=remote(ssh,"import json,pathlib,sys;print((pathlib.Path(sys.argv[1])/'runtime_receipt.json').read_text())",[runtime])
    assert check['checkpoint_resume_exact'] and not check['research_training']
    assert check['torch'].split('+')[0]=='2.12.0' and check['numpy']=='2.4.6'
    cfg,ident,data,jobs,oid,pid,fits,actions=run.load()
    target=str(Path(runtime).parent/cfg['name'])
    paths=closure(run.ROOT,['scripts.train_m3w_easy_hurdle_portable'])
    bindings={str(p.relative_to(run.ROOT)):run.base.digest(p) for p in paths}
    bundle=io.BytesIO()
    with tarfile.open(fileobj=bundle,mode='w:gz') as tar:
        for path in paths:
            raw=path.read_bytes();entry=tarfile.TarInfo(str(path.relative_to(run.ROOT)));entry.size=len(raw);entry.mode=0o644
            tar.addfile(entry,io.BytesIO(raw))
    code=r'''
import io,json,pathlib,sys,tarfile,hashlib
root=pathlib.Path(sys.argv[1]); assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
root.mkdir(exist_ok=True); owner=root/'.owner.json'; expected={'experiment':'european_easy_hurdle_v1','registration_sha256':sys.argv[2]}
if owner.exists(): assert json.loads(owner.read_text())==expected
else: owner.write_text(json.dumps(expected)+'\n')
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
    checks={}
    for m in tar.getmembers():
        assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
        p=root/'code'/m.name; content=tar.extractfile(m).read(); p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists(): assert p.read_bytes()==content
        else: p.write_bytes(content)
        checks[m.name]=hashlib.sha256(content).hexdigest()
print(json.dumps({'bindings':checks}))
'''
    out=remote(ssh,code,[target,run.base.digest(run.PUBLIC/'registration.json')],bundle.getvalue())
    assert out['bindings']==bindings
    refs=[]
    for c in run.base.floor_api.contexts({k:data[k] for k in run.parent.CAUSAL_KEYS},jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';assert re.fullmatch(r'[A-Za-z0-9_]+',name)
            path=run.PRIVATE/'remote_inputs'/(name+'.json')
            if path.exists() and not a.resume:
                raise ValueError('Existing transfer receipt requires resume')
            fit,ids,u,y,pr,fid=run.fitting(c,data,pair,fits[name],pid)
            pr_scalars={k:pr[k] for k in ('clip','cost_scale','training_sites')}
            buf=io.BytesIO()
            np.savez_compressed(buf,x=c['x'][fit],u=u,y=y,env=c['env'][fit],ids=ids,
                sites=data['sites'][ids],recordings=data['recordings'][ids],frames=data['frames'][ids],
                **{'pr_'+k:pr[k] for k in ('known','weights','mean','std')},
                identity_json=np.array(json.dumps(fid)),preprocess_json=np.array(json.dumps(pr_scalars)))
            payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest()
            code=r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); assert json.loads((root/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
name=sys.argv[2]; assert name.replace('_','').isalnum()
content=sys.stdin.buffer.read(); assert hashlib.sha256(content).hexdigest()==sys.argv[3]
p=root/'inputs'/(name+'.npz');p.parent.mkdir(exist_ok=True)
if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3]
else:
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f: f.write(content);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
print(json.dumps({'group':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}))
'''
            ref=remote(ssh,code,[target,name,sha],payload);assert ref==dict(group=name,sha256=sha,bytes=len(payload))
            run.base.immutable_json(path,ref);refs.append(ref)
            run.beat('fitting_packet_streamed',group=name,complete=len(refs),bytes=len(payload))
    assert len(refs)==108
    manifest=dict(registration=ident,code_bindings=bindings,groups=refs,held_rows_transferred=False,
                  input_role='head_fitting_sources_only',remote_runtime=check)
    code=r'''
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); a=json.loads(sys.stdin.read())
for name,value in a.items():
    assert name in ('input_manifest.json','config.json');p=root/name
    if p.exists(): assert json.loads(p.read_text())==value
    else: p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'complete':True}))
'''
    remote(ssh,code,[target],json.dumps({'input_manifest.json':manifest,'config.json':cfg}).encode())
    run.base.immutable_json(run.PRIVATE/'remote_input_manifest.json',dict(remote_path=target,**manifest))
    run.base.immutable_json(run.PUBLIC/'remote_input_transfer.json',dict(groups=108,bytes=sum(r['bytes'] for r in refs),
        code_bindings=bindings,groups_manifest=refs,held_rows_transferred=False,independent_roles_read=False,
        result_source='fresh_run_fitting_only_streamed_export',large_local_temporary_files=False))


if __name__=='__main__':
    main()
