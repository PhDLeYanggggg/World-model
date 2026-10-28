"""Execute the unchanged optimizer with an explicit CREATE path-alias repair."""
import argparse
import json
from pathlib import Path
import subprocess

from scripts import submit_m3w_task_risk_stream as stream

original, base = stream.original, stream.base
AMENDMENT = original.PUBLIC/'path_amendment.json'


def register():
    stream.register()
    files = [Path(__file__).resolve(), base.ROOT/'scripts/train_m3w_task_risk_portable.py',
             base.ROOT/'tests/test_m3w_task_risk_port.py', original.PUBLIC/'path_repair.md']
    value = dict(training_registration_sha256=base.sha(original.REG),
        transport_amendment_sha256=base.sha(stream.AMENDMENT),
        bindings={str(p.relative_to(base.ROOT)):base.sha(p) for p in files},
        change='logical owned root comparison; retain canonical code containment',
        train_function='unchanged_original_import', numerical_code_changed=False,
        registration_changed=False, training_started_before_repair=False,
        submission_intents_before_repair=0, objective_changed=False,
        independent_roles_read=False, deployment_changed=False)
    base.once(AMENDMENT, value)
    return value


def remote_code(kind):
    code = original.SUBMIT if kind == 'submit' else original.OBSERVE
    code = code.replace('from scripts.train_m3w_task_risk import ',
                        'from scripts.train_m3w_task_risk_portable import ')
    code = code.replace(' -m scripts.train_m3w_task_risk ', ' -m scripts.train_m3w_task_risk_portable ')
    return code.replace('root,source,cfg,reg,manifest=verify(root);',
        'from scripts.train_m3w_task_risk_portable import check_port\ncheck_port(root)\nroot,source,cfg,reg,manifest=verify(root);')


PORT_INSTALL = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_task_risk_v1')
p=json.loads(sys.stdin.buffer.read());a=json.loads(p['amendment'])
assert hashlib.sha256((root/'registration.json').read_bytes()).hexdigest()==a['training_registration_sha256']
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==a['training_registration_sha256']
def exact(path,raw):
 assert not path.is_symlink()
 if path.exists():assert path.read_bytes()==raw
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(raw)
for rel,raw in p['files'].items():
 assert rel in a['bindings'] and hashlib.sha256(raw.encode()).hexdigest()==a['bindings'][rel]
 target=root/'code'/rel;assert target.resolve().is_relative_to((root/'code').resolve())
 exact(target,raw.encode())
assert set(p['files'])==set(a['bindings'])
exact(root/'path_amendment.json',p['amendment'].encode())
print(json.dumps(dict(path_amendment_verified=True,optimizer_changed=False)))
'''


def submit(phase, resume=False):
    amend = register(); reg = original.register()
    for rel in [*reg['bindings'], *amend['bindings'], *stream.register()['bindings'],
                str(original.REG.relative_to(base.ROOT)),str(stream.AMENDMENT.relative_to(base.ROOT)),
                str(AMENDMENT.relative_to(base.ROOT))]:
        p = subprocess.run(['git','show','HEAD:'+rel],cwd=base.ROOT,capture_output=True)
        if p.returncode or p.stdout != (base.ROOT/rel).read_bytes():
            raise ValueError('Commit all original and amended bindings first')
    base.remote(stream.installer(), [original.REMOTE,base.REMOTE], stream.payload(reg),timeout=90)
    payload = dict(amendment=AMENDMENT.read_text(), files={r:(base.ROOT/r).read_text() for r in amend['bindings']})
    port = base.remote(PORT_INSTALL,[original.REMOTE],json.dumps(payload).encode(),timeout=60)
    result = base.remote(remote_code('submit'),[original.REMOTE,base.RUNTIME,phase,
        base.sha(original.REG),'resume' if resume else 'new'],timeout=90)
    base.once(original.PUBLIC/(phase+'_attempt'+str(result['attempt'])+'_submission.json'),result)
    return dict(path_port=port,submission=result)


def observe(phase, collect=False):
    out = base.remote(remote_code('observe'),[original.REMOTE,phase,'collect' if collect else 'status'],timeout=90)
    if collect:
        for rel,raw in out.pop('files').items():
            p = (original.PUBLIC/rel).resolve()
            if not p.is_relative_to(original.PUBLIC.resolve()): raise ValueError('Owned metadata only')
            base.once(p,json.loads(raw))
        base.once(original.PUBLIC/(phase+'_collection.json'),out)
    else:
        from datetime import datetime,timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        base.once(original.PUBLIC/(phase+'_observation_'+stamp+'.json'),out)
    if out.get('artifact') and 'fits' in out['artifact']:
        out={**out,'artifact':{k:v for k,v in out['artifact'].items() if k!='fits'}}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','pilot','train','resume','status','collect'])
    p.add_argument('--phase',choices=['pilot','train'],default='pilot');a=p.parse_args()
    if a.mode=='register':
        register();out=dict(amendment_sha256=base.sha(AMENDMENT))
    elif a.mode in ('status','collect'):out=observe(a.phase,a.mode=='collect')
    else:out=submit(a.phase if a.mode=='resume' else a.mode,a.mode=='resume')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
