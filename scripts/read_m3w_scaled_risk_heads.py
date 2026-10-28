"""Admit all final scaled/control heads plus explicitly cached paired controls."""
import json

from scripts import run_m3w_scaled_risk as training
from scripts import read_m3w_task_risk_final_heads as previous

ROOT=training.ROOT
Reader=previous.Reader
NEURAL_ARMS=('quadratic','task_utility','unit_control','budget_scaled')


def admission():
    reg,cfg=training.admission(); olddocs,oldinv=previous.local_admission()
    freeze=json.loads((training.PUBLIC/'training_freeze.json').read_text())
    if (freeze['neural_fits']!=144 or freeze['exact_previous_control_replays']!=72
            or len(freeze['fits'])!=144 or freeze['registration_sha256']!=training.base.sha(training.REG)
            or freeze['validation_scored'] or freeze['independent_roles_read']):
        raise ValueError('All144 fixed finals with exact unit controls required')
    expected={(r['group'],r['source'],r['seed'],mode) for r in reg['expected'] for mode in cfg['modes']}
    docs={}; inventory={}
    for ref in freeze['fits']:
        p=ROOT/ref['path']
        if (p.is_symlink() or not p.resolve().is_relative_to((training.PUBLIC/'fits').resolve())
                or training.base.sha(p)!=ref['sha256']):raise ValueError('Changed fit receipt')
        doc=json.loads(p.read_text());i=doc['identity'];key=i['group'],i['source'],i['seed'],doc['mode']
        if (key not in expected or key in docs or doc['step']!=2000 or doc['validation_scored']
                or doc['independent_roles_read'] or doc['registration_sha256']!=freeze['registration_sha256']
                or doc['unit_control_replay_exact']!=(doc['mode']=='unit_control')):
            raise ValueError('Wrong final fit identity')
        cp=doc['checkpoint'];path=ROOT/cp['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((training.PRIVATE/'heads').resolve())
                or path.stat().st_size!=cp['bytes'] or training.base.sha(path)!=cp['sha256']):
            raise ValueError('Changed model payload')
        docs[key]=doc;inventory[str(path)]=cp
    if set(docs)!=expected:raise ValueError('Missing scaling control')
    from src.world_model import m3w_scaled_risk_training as api
    for key,doc in list(docs.items()):
        if key[3]=='unit_control':
            new=api.old.core.read_checkpoint(ROOT/doc['checkpoint']['path'])
            prior=olddocs[*key[:3],'task_risk']
            api.assert_control(new,api.old.core.read_checkpoint(ROOT/prior['checkpoint']['path']))
    for key,doc in olddocs.items():
        if key[3] in ('quadratic','task_utility'):
            docs[key]=doc;path=str(ROOT/doc['checkpoint']['path']);inventory[path]=oldinv['checkpoints'][path]
    if len(docs)!=288:raise ValueError('144new plus144cached controls required')
    return docs,dict(training_freeze_sha256=training.base.sha(training.PUBLIC/'training_freeze.json'),
        parent_training_freeze_sha256=oldinv['training_freeze_sha256'],checkpoints=inventory)
