"""Register and stream a source-only calibration control to owned CREATE space."""
import argparse
import io
import json
from pathlib import Path
import platform
import shlex
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 environment')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_selected_pool_accounting as parent
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote, HANDOFF
from scripts.manage_m3w_boundary_diagnostic import RECEIVER
from scripts.export_m3w_easy_hurdle_create import closure
from scripts.m3w_packet_stream import PacketStream

NAME = 'european_selected_set_calibration_v1'
PUBLIC = parent.PUBLIC.parent/NAME
PRIVATE = parent.PRIVATE.parent/NAME
REMOTE = '/users/k24101830/m3w/'+NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
digest, immutable = parent.digest, parent.immutable


def registration():
    parent.registration()
    files = closure(ROOT,['scripts.run_m3w_selected_set_calibration'])
    controls = [Path(__file__),CONFIG,PUBLIC/'protocol.md',PUBLIC/'asset_and_method_note.md',
        ROOT/'tests/test_m3w_selected_set_calibration.py',
        ROOT/'tests/test_m3w_selected_set_runner.py',
        ROOT/'scripts/m3w_packet_stream.py',ROOT/'scripts/manage_m3w_boundary_diagnostic.py',
        ROOT/'scripts/manage_m3w_easy_gradient_diagnostic.py']
    return dict(name=NAME,config_sha256=digest(CONFIG),
        parent_registration_sha256=digest(parent.PUBLIC/'registration.json'),
        parent_component_verification_sha256=digest(parent.parent.PUBLIC/'verification.json'),
        remote_code={str(p.relative_to(ROOT)):digest(p) for p in files},
        controls={str(p.relative_to(ROOT)):digest(p) for p in controls},
        source_only=True,transfer_evaluated=False,new_neural_updates=0,independent_roles_read=False)


def remote(code,payload):
    got = call_remote(code,payload)
    if got['returncode'] != 0:
        raise RuntimeError(json.dumps(got))
    return json.loads(got['stdout'])


def export(reg):
    setup = r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['home'])
assert r==pathlib.Path('/users/k24101830/m3w/european_selected_set_calibration_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
r.mkdir(exist_ok=True)
assert not (r/'submission_intent.json').exists(),'Inputs frozen after submission intent'
def once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert path.read_text()==text
    else:
        with path.open('x') as f: f.write(text)
once(r/'.owner.json',json.dumps({'experiment':p['name'],'project':'M3W'})+'\n')
once(r/'registration.json',p['registration_text']);once(r/'config.json',p['config_text'])
for name,text in p['files'].items():
    assert name in p['registration']['remote_code']
    assert hashlib.sha256(text.encode()).hexdigest()==p['registration']['remote_code'][name]
    assert not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts
    once(r/'code'/name,text)
print(json.dumps({'prepared':True,'existing_packets':len(list((r/'inputs').glob('*.npz')))}))
'''
    print(json.dumps(remote(setup,dict(home=REMOTE,name=NAME,registration=reg,
        registration_text=(PUBLIC/'registration.json').read_text(),config_text=CONFIG.read_text(),
        files={p:(ROOT/p).read_text() for p in reg['remote_code']}))),flush=True)
    cfg = json.loads(CONFIG.read_text())
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    stream = PacketStream.__new__(PacketStream)
    stream.command = ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,REMOTE,NAME])]
    stream.process = None
    parent.core.torch.set_num_threads(cfg['cpu_threads'])
    parent.core.torch.set_num_interop_threads(1)
    _,_,data,jobs,oid,_,_,_ = parent.inner.old.load()
    docs, originals = parent.parent.docs(), parent.parent.parent.docs()
    refs, total, began = [], 0, time.monotonic()
    with stream:
        for c in parent.parent.parent.contexts(data,jobs,oid):
            for site in parent.inner.sources(c):
                group = c['name']+'_fit_'+site
                at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c,data,site)
                _, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
                vid = ids[val]
                for seed in parent.parent.parent.api.SEEDS:
                    old, doc = originals[group,seed], docs[group,seed]
                    assert old['partition'] == doc['identity']['partition'] == partition
                    path = ROOT/old['checkpoint']['path']
                    assert parent.base.artifact(path) == old['checkpoint']
                    state = joblib.load(path)
                    assert state['identity']['upstream'] == upstream
                    p, support = parent.forest.api.predict(state,x[val],env[val])
                    assert parent.base.inter.array_hash(p) == doc['identity']['prediction_hash']
                    assert parent.base.inter.array_hash(y[val]) == doc['identity']['target_hash']
                    arrays = dict(p=p,y=y[val],env=env[val],moving=c['moving'][at][val],
                        support=support,recordings=data['recordings'][vid].astype(str))
                    raw = parent.parent.api.eligible(p,arrays['moving'],support)
                    assert parent.base.inter.array_hash(raw) == old['validation']['action_hashes']['forest']
                    meta = dict(source=site,identity=dict(group=group,head_seed=seed,
                        source=site,partition=partition,checkpoint=old['checkpoint'],
                        ids_hash=parent.base.inter.array_hash(vid),registration_sha256=digest(PUBLIC/'registration.json')),
                        array_hashes={k:parent.base.inter.array_hash(v) for k,v in arrays.items()},
                        parent_oof_action_hashes=doc['oof_action_hashes'],
                        parent_source=doc['source'],parent_final=doc['final'])
                    arrays['meta_json'] = np.array(json.dumps(meta,sort_keys=True))
                    buf = io.BytesIO();np.savez_compressed(buf,**arrays);packet = buf.getvalue()
                    total += len(packet)
                    assert total <= cfg['input_byte_cap']
                    refs.append(stream.send(group+'_head'+str(seed),packet))
                    if len(refs) == 1 or len(refs)%12 == 0:
                        print(json.dumps(dict(phase='source_input_stream',groups=len(refs),bytes=total,seconds=time.monotonic()-began)),flush=True)
    assert len(refs) == cfg['source_heads']
    manifest = dict(packets=refs,bytes=total,source_only=True,
        registration_sha256=digest(PUBLIC/'registration.json'))
    seal = r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['home'])
assert json.loads((r/'.owner.json').read_text())['experiment']==p['name']
assert not (r/'submission_intent.json').exists()
for e in p['manifest']['packets']:
    f=r/'inputs'/(e['group']+'.npz')
    assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256']
text=json.dumps(p['manifest'],indent=2)+'\n';f=r/'input_manifest.json'
if f.exists(): assert f.read_text()==text
else: f.write_text(text)
print(json.dumps({'verified_source_packets':len(p['manifest']['packets'])}))
'''
    print(json.dumps(remote(seal,dict(home=REMOTE,name=NAME,manifest=manifest))))
    immutable(PUBLIC/'input_manifest.json',manifest)
    immutable(PUBLIC/'export_receipt.json',dict(groups=len(refs),bytes=total,
        seconds=time.monotonic()-began,local_numeric_cache=False,remote_calibration='not_run'))


def submit():
    path = PRIVATE/'submission.json'
    assert not path.exists(),'Inspect existing/uncertain submission; do not duplicate'
    script = '''#!/bin/bash -l
#SBATCH --job-name=m3w_selected_set_v1
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=calibration-%j.out
#SBATCH --error=calibration-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/run_m3w_selected_set_calibration.py --home "$PWD"
'''
    code = r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['home'])
assert r==pathlib.Path('/users/k24101830/m3w/european_selected_set_calibration_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']==p['name']
assert hashlib.sha256((r/'input_manifest.json').read_bytes()).hexdigest()==p['manifest_hash']
assert not (r/'submission_intent.json').exists() and not (r/'submission_receipt.json').exists()
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and 'm3w_selected_set_v1' not in q.stdout
(r/'run.sh').write_text(p['script'])
(r/'submission_intent.json').write_text(json.dumps({'manifest':p['manifest_hash'],'queue':q.stdout})+'\n')
z=subprocess.run(['sbatch','--parsable','run.sh'],cwd=r,capture_output=True,text=True,timeout=30)
out=dict(returncode=z.returncode,stdout=z.stdout,stderr=z.stderr)
if z.returncode==0:
    out['job_id']=z.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
(r/'submission_receipt.json').write_text(json.dumps(out)+'\n')
print(json.dumps(out))
'''
    result = call_remote(code,dict(home=REMOTE,name=NAME,manifest_hash=digest(PUBLIC/'input_manifest.json'),script=script))
    immutable(path,result)
    if result['returncode'] != 0:
        raise RuntimeError('Uncertain submission: inspect remote intent; never duplicate')
    receipt = json.loads(result['stdout']);immutable(PUBLIC/'submission_receipt.json',receipt)
    assert receipt['returncode'] == 0
    print(json.dumps(receipt))


def inspect_remote():
    code = r'''
import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path(json.loads(sys.stdin.read())['home']);out={}
assert r==pathlib.Path('/users/k24101830/m3w/european_selected_set_calibration_v1')
for name in ('submission_receipt.json','heartbeat.json','complete.json','summary.json'):
    p=r/name
    if p.exists(): out[name]=json.loads(p.read_text())
if 'submission_receipt.json' in out:
    j=out['submission_receipt.json'].get('job_id')
    if j:
        q=subprocess.run(['sacct','-j',j,'-X','--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=25)
        out['accounting']={'returncode':q.returncode,'stdout':q.stdout}
if 'complete.json' in out:
    d=out['complete.json'];assert d['exact_replay'] and d['source_heads']==72
    assert hashlib.sha256((r/'summary.json').read_bytes()).hexdigest()==d['summary_sha256']
    for e in d['groups']:
        p=r/'groups'/(e['group']+'.json')
        assert p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
    out['remote_artifact_hashes_verified']=True
print(json.dumps(out))
'''
    out = remote(code,dict(home=REMOTE))
    print(json.dumps(out,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','export','submit','inspect'])
    a = p.parse_args()
    if a.phase == 'inspect': inspect_remote();return
    reg = registration()
    if a.phase == 'register':
        immutable(PUBLIC/'registration.json',reg);print('Registered source-only calibration');return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    parent.base.inter.committed(PUBLIC/'registration.json')
    if a.phase == 'export': export(reg)
    else: submit()


if __name__ == '__main__':
    main()
