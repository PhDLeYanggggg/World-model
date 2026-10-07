"""Complete144-head admission under the explicit54+18 reference contract."""
import hashlib
import json
from pathlib import PurePosixPath

from scripts import read_m3w_easy_harm_final_heads as prior
from scripts import train_m3w_easy_harm_recovery_v2 as recovery

manager, ROOT, HOME, HEADS, Reader = prior.manager, prior.ROOT, prior.HOME, prior.HEADS, prior.Reader


def validate(freeze, files, expected, cfg, registration_sha, amendment, amendment_sha):
    required=dict(neural_fits=144,source_heads=72,registration_sha256=registration_sha,
        original_implementation_controls_exact=72,historical_quadratic_controls_exact=54,
        historical_nonexact_replay_verified=18,historical_bitwise_reproduction_complete=False,
        control_execution_amendment_sha256=amendment_sha,
        previous_amendment_sha256=amendment['previous_amendment_sha256'],
        preserved_accepted_fits=110,adopted_diagnostic_quadratic_fits=17,new_candidate_fits=17)
    for k,v in required.items():
        if freeze.get(k)!=v:
            raise ValueError('Complete54+18 amended grid required: '+k)
    for k in ('validation_scored','independent_roles_read','new_forecasters','deployment_changed','stage5c_executed','smc_enabled'):
        if freeze.get(k) is not False:
            raise ValueError('Unscored non-deployed grid required')
    wanted={(r['group'],r['source'],r['seed'],a) for r in expected for a in cfg['arms']}
    if len(expected)!=72 or len(wanted)!=144 or set(cfg['arms'])!={'quadratic','easy_deviance'}:
        raise ValueError('Original72 identities and two arms required')
    if len(amendment['references'])!=18 or len(amendment['preserved_fit_refs'])!=110:
        raise ValueError('Frozen18 references and110 receipt preservation required')
    counts=dict(historical=0,replay=0,candidate=0); result={};paths=set()
    for ref in freeze['fits']:
        path=ref['path']; rel=PurePosixPath(path)
        if (rel.is_absolute() or '..' in rel.parts or not rel.is_relative_to(HOME+'/fits')
                or str(rel)!=path or path in paths or path not in files):
            raise ValueError('Unique owned receipt required')
        raw=files[path]
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise ValueError('Changed receipt')
        row=json.loads(raw); identity=row['identity'];cp=row['checkpoint']
        key=identity['group'],identity['source'],identity['seed'],row['arm']
        name=identity['group']+'_head'+str(identity['seed'])+'_'+row['arm']
        if key not in wanted or key in result or rel.stem!=name:
            raise ValueError('Unknown or repeated fit identity')
        if cp['path']!=HEADS+'/'+name+'/checkpoint.pt.gz' or not 0<cp['bytes']<=2*2**20:
            raise ValueError('Wrong checkpoint path/size')
        if len(cp['sha256'])!=64 or any(c not in '0123456789abcdef' for c in cp['sha256']):
            raise ValueError('Invalid checkpoint hash')
        counts[recovery.validate_fit(row,name,amendment,amendment['previous_amendment_sha256'],amendment_sha)]+=1
        result[key]=row;paths.add(path)
    if set(result)!=wanted or paths!=set(files) or counts!=dict(historical=54,replay=18,candidate=72):
        raise ValueError('Incomplete or extra final fits')
    for ref in amendment['preserved_fit_refs']:
        if ref['path'] not in files or hashlib.sha256(files[ref['path']]).hexdigest()!=ref['sha256']:
            raise ValueError('Previously accepted receipt changed')
    return result


def local_admission():
    home=manager.PUBLIC
    freeze=json.loads((home/'training_freeze.json').read_text())
    reg=json.loads(manager.REG.read_text());cfg=json.loads(manager.CONFIG.read_text())
    amendment=json.loads((home/recovery.AMENDMENT).read_text())
    old=json.loads((home/'control_execution_amendment.json').read_text())
    if (manager.base.sha(manager.CONFIG)!=reg['config_sha256']
            or amendment['training_registration_sha256']!=manager.base.sha(manager.REG)
            or amendment['previous_amendment_sha256']!=manager.base.sha(home/'control_execution_amendment.json')):
        raise ValueError('Changed training contract')
    for rel,digest in {**reg['bindings'],**old['bindings'],**amendment['bindings']}.items():
        if manager.base.sha(ROOT/rel)!=digest:
            raise ValueError('Changed registered implementation')
    for ref in amendment['proofs']:
        if manager.base.sha(ROOT/ref['path'])!=ref['sha256']:
            raise ValueError('Changed reference evidence')
    files={}
    for ref in freeze['fits']:
        path=(ROOT/ref['path']).resolve()
        if not path.is_relative_to(home.resolve()/'fits'):
            raise ValueError('Misplaced fit metadata')
        files[ref['path']]=path.read_bytes()
    documents=validate(freeze,files,reg['expected'],cfg,manager.base.sha(manager.REG),amendment,
                       manager.base.sha(home/recovery.AMENDMENT))
    collection=json.loads((home/'train_collection.json').read_text())
    if collection['artifact']!=freeze:
        raise ValueError('Collected grid differs')
    job=collection['join']['submission']['job_id']
    rows=[line.split('|') for line in collection['join']['accounting']['stdout'].splitlines()]
    if not any(len(r)>=4 and r[0]==job and r[1]=='COMPLETED' and r[3]=='0:0' for r in rows):
        raise ValueError('Successful final join required')
    inventory=dict(training_freeze_sha256=manager.base.sha(home/'training_freeze.json'),
                   checkpoints={row['checkpoint']['path']:row['checkpoint'] for row in documents.values()})
    return documents,inventory
