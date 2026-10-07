"""New admission provenance, identical pre-registered six-arm calculations."""
import argparse
import fcntl
import json
import platform

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 before numerical imports')

from scripts import run_m3w_easy_harm_readout as original
from scripts import read_m3w_easy_harm_final_heads_v2 as heads

ROOT=heads.ROOT
PUBLIC=heads.manager.PUBLIC/'readout_verified_v2'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/readout_verified_v2'


def registration():
    old_home=heads.manager.PUBLIC/'readout'
    calculation=json.loads((old_home/'calculation_registration.json').read_text())
    prior=json.loads((old_home/'registration.json').read_text())
    for rel,digest in {**calculation['bindings'],**prior['bindings']}.items():
        if original.sha(ROOT/rel)!=digest:
            raise ValueError('Original calculation or runner changed')
    files=original.parent.forest.closure(ROOT,['scripts.run_m3w_easy_harm_readout_v2'])
    files += [ROOT/'tests/test_m3w_easy_harm_final_heads_v2.py',PUBLIC/'admission_protocol.md']
    train=json.loads(heads.manager.REG.read_text())
    reg=dict(bindings={str(p.relative_to(ROOT)):original.sha(p) for p in sorted(set(files))},
        training_registration_sha256=original.sha(heads.manager.REG),
        execution_amendment_sha256=original.sha(heads.manager.PUBLIC/heads.recovery.AMENDMENT),
        original_readout_registration_sha256=original.sha(old_home/'registration.json'),
        calculation_registration_sha256=original.sha(old_home/'calculation_registration.json'),
        controls_registration_sha256=original.sha(original.controls.PUBLIC/'registration.json'),
        expected=[dict(group=r['group'],source=r['source'],head_seed=r['seed']) for r in train['expected']],
        arms=list(original.api.ARMS),new_validation_predictions=False,independent_roles_read=False,
        checkpoint_selection=False,threshold_search=False,risk_budget=.02,
        scientific_calculation_unchanged=True,historical_controls_exact=54,original_replay_controls_exact=18)
    return json.loads(heads.manager.CONFIG.read_text()),reg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','preflight','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg,reg=registration()
    if a.phase=='register':
        original.once(PUBLIC/'registration.json',reg)
        print(json.dumps(dict(registered=True,actual_readout='not_run',bindings=len(reg['bindings']))));return
    if reg!=json.loads((PUBLIC/'registration.json').read_text()):
        raise ValueError('Readout binding drift')
    original.parent.base.inter.committed(PUBLIC/'registration.json')
    docs,inventory=heads.local_admission()
    original.parent.base.inter.committed(heads.manager.PUBLIC/'training_freeze.json')
    if a.phase=='preflight':
        print(json.dumps(dict(complete_heads=len(docs),actual_readout='not_run')));return
    if (PUBLIC/'complete.json').exists():
        raise RuntimeError('Completed readout is immutable')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    original.fitting.torch.set_num_threads(4);original.fitting.torch.set_num_interop_threads(1)
    # Only artifact destinations and the explicit complete-grid admission change.
    original.PUBLIC,original.PRIVATE,original.heads=PUBLIC,PRIVATE,heads
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.run(cfg,reg,docs,inventory,a.resume)


if __name__=='__main__':
    main()
