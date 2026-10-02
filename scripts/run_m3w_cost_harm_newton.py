"""Solver-only amendment for the registered squared-cost experiment."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_cost_aligned_positive_harm as runner
from src.world_model import m3w_cost_harm_newton as api

NAME = 'european_cost_harm_newton_v1'
PUBLIC, PRIVATE = runner.PUBLIC.parent/NAME, runner.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
OLD = runner.PUBLIC
REMOTE = '/users/k24101830/m3w/'+NAME


def registration():
    old = json.loads((OLD/'registration.json').read_text())
    for path, digest in old['bindings'].items(): assert runner.sha(ROOT/path) == digest
    assert runner.diagnostic.registration() == json.loads((runner.diagnostic.PUBLIC/'registration.json').read_text())
    probe = json.loads((OLD/'solver_newton_diagnosis.json').read_text())
    assert probe['validation_evaluated'] is False and probe['events'][-1]['status'] == 'converged'
    paths = runner.parent.forest.closure(ROOT, ['scripts.run_m3w_cost_harm_newton'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_cost_harm_newton.py',
              OLD/'solver_diagnosis.json', OLD/'solver_newton_diagnosis.json']
    return dict(bindings={str(p.relative_to(ROOT)): runner.sha(p) for p in paths},
        original_registration_sha256=runner.sha(OLD/'registration.json'), source_heads=72,
        amendment='Exact residual curvature; objective/data/tolerance/initialization unchanged',
        independent_roles_read=False, new_neural_updates=0, new_tree_splits=0)


def storage(refs=None):
    code = r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_cost_harm_newton_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and shutil.disk_usage(r.parent).free>p['cap']
r.mkdir(exist_ok=True)
for name,txt in {'.owner.json':json.dumps(dict(experiment=p['name'],project='M3W'))+'\n',
                 'registration.json':p['registration'],'config.json':p['config']}.items():
    f=r/name
    if f.exists():assert f.read_text()==txt
    else:
        with f.open('x') as out:out.write(txt)
existing=[]
for f in sorted((r/'inputs').glob('*.npz')):
    existing.append(dict(group=f.stem,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
if p['refs'] is not None:assert sorted(existing,key=lambda x:x['group'])==sorted(p['refs'],key=lambda x:x['group'])
print(json.dumps(dict(existing=existing,owned_root_verified=True,personal_quota='unknown',
    remote_science_executed=False,m3w_jobs=[l for l in q.stdout.splitlines() if 'm3w' in l.lower()])))
'''
    cfg = json.loads(CONFIG.read_text())
    return runner.prior.prior.leaf.previous.independent.read_remote(code, dict(root=REMOTE, name=NAME,
        cap=cfg['remote_checkpoint_cap_bytes'], registration=(PUBLIC/'registration.json').read_text(),
        config=CONFIG.read_text(), refs=refs))


def main():
    runner.api, runner.NAME, runner.PUBLIC, runner.PRIVATE = api, NAME, PUBLIC, PRIVATE
    runner.CONFIG, runner.REMOTE, runner.registration, runner.storage = CONFIG, REMOTE, registration, storage
    runner.main()


if __name__ == '__main__': main()
