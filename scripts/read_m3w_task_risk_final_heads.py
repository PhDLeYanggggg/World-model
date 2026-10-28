"""Hash-verified local final heads; never substitute pilot or missing fits."""
import json
from pathlib import Path

from scripts import manage_m3w_task_risk as manager
from scripts import run_m3w_task_risk_local as execution

ROOT=manager.base.ROOT


def local_admission():
    reg,cfg=execution.check_registration()
    freeze=json.loads((manager.PUBLIC/'training_freeze.json').read_text())
    if (freeze['neural_fits']!=216 or freeze['source_heads']!=72 or len(freeze['fits'])!=216
            or freeze['registration_sha256']!=manager.base.sha(manager.REG)
            or freeze['local_amendment_sha256']!=manager.base.sha(execution.AMENDMENT)
            or freeze['validation_scored'] or freeze['independent_roles_read']):
        raise ValueError('Complete frozen TRAIN grid required')
    expected={(r['group'],r['source'],r['seed'],arm) for r in reg['expected'] for arm in cfg['arms']}
    docs={};checks={}
    for ref in freeze['fits']:
        p=ROOT/ref['path']
        if p.is_symlink() or not p.resolve().is_relative_to((manager.PUBLIC/'fits').resolve()):
            raise ValueError('Only owned fit receipts')
        if manager.base.sha(p)!=ref['sha256']:raise ValueError('Changed fit receipt')
        doc=json.loads(p.read_text());i=doc['identity'];key=i['group'],i['source'],i['seed'],doc['arm']
        if (key not in expected or key in docs or doc['step']!=2000 or doc['validation_scored']
                or doc['independent_roles_read'] or doc['registration_sha256']!=freeze['registration_sha256']
                or doc['local_amendment_sha256']!=freeze['local_amendment_sha256']):
            raise ValueError('Wrong final fit identity or provenance')
        cp=doc['checkpoint'];path=ROOT/cp['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((execution.PRIVATE/'heads').resolve())
                or path.stat().st_size!=cp['bytes'] or manager.base.sha(path)!=cp['sha256']):
            raise ValueError('Changed final checkpoint payload')
        docs[key]=doc;checks[str(path)]=cp
    if set(docs)!=expected:raise ValueError('Missing final identity')
    return docs,dict(training_freeze_sha256=manager.base.sha(manager.PUBLIC/'training_freeze.json'),checkpoints=checks)


class Reader:
    def __init__(self,inventory):
        self.inventory=inventory;self.fetched_bytes=0

    def __enter__(self):return self

    def __exit__(self,*args):return False

    def load(self,path):
        from src.world_model import m3w_task_risk_training as training
        path=Path(path);ref=self.inventory['checkpoints'][str(path)]
        if path.stat().st_size!=ref['bytes'] or manager.base.sha(path)!=ref['sha256']:
            raise ValueError('Checkpoint changed after admission')
        self.fetched_bytes+=ref['bytes']
        return training.core.read_checkpoint(path)
