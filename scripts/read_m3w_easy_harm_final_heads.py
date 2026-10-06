"""Complete-grid admission and owned, cache-free final cost-head access."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

from scripts import manage_m3w_easy_harm_deviance as manager
from scripts import read_m3w_temporal_auxiliary_create as transport

ROOT = manager.base.ROOT
HOME = str(manager.PUBLIC.relative_to(ROOT))
HEADS = 'data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/heads'


def validate(freeze, files, expected, cfg, registration_sha, amendment):
    if (freeze.get('neural_fits') != 144 or freeze.get('source_heads') != 72
            or freeze.get('registration_sha256') != registration_sha
            or freeze.get('original_implementation_controls_exact') != 72
            or freeze.get('historical_quadratic_controls_exact') != 71
            or freeze.get('historical_nonexact_replay_verified') != 1
            or freeze.get('historical_bitwise_reproduction_complete') is not False):
        raise ValueError('Complete amended grid and honest historical counts required')
    for key in ('validation_scored','independent_roles_read','new_forecasters',
                'deployment_changed','stage5c_executed','smc_enabled'):
        if freeze.get(key) is not False:
            raise ValueError('Unscored, non-deployed fixed-final training required')
    wanted = {(r['group'],r['source'],r['seed'],a) for r in expected for a in cfg['arms']}
    if len(wanted) != 144 or len(expected) != 72 or set(cfg['arms']) != {'quadratic','easy_deviance'}:
        raise ValueError('Original registered 72-pair identities required')
    result, paths, historical, replayed = {}, set(), 0, 0
    for ref in freeze['fits']:
        path = ref['path']; relative = PurePosixPath(path)
        if (relative.is_absolute() or '..' in relative.parts or not relative.is_relative_to(HOME+'/fits')
                or str(relative) != path or path in paths or path not in files):
            raise ValueError('Unique owned training receipt required')
        raw = files[path]
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('Changed training receipt')
        row = json.loads(raw); identity = row['identity']; cp = row['checkpoint']
        key = identity['group'],identity['source'],identity['seed'],row['arm']
        if key not in wanted or key in result or row['step'] != 2000 or row['validation_scored'] is not False:
            raise ValueError('Unique complete unscored registered head required')
        name = identity['group']+'_head'+str(identity['seed'])+'_'+row['arm']
        if cp['path'] != HEADS+'/'+name+'/checkpoint.pt.gz' or not 0 < cp['bytes'] <= 2*2**20:
            raise ValueError('Checkpoint identity or size invalid')
        if len(cp['sha256']) != 64 or any(c not in '0123456789abcdef' for c in cp['sha256']):
            raise ValueError('Valid frozen checkpoint digest required')
        if row['arm'] == 'quadratic':
            if row['parent_control_exact']:
                historical += 1
            else:
                if (identity != amendment['exception_identity']
                        or row.get('historical_control_exact') is not False
                        or row.get('original_implementation_control_exact') is not True
                        or row.get('floating_tolerance_relaxed') is not False
                        or row.get('reference_checkpoint') != amendment['reference_checkpoint']
                        or row.get('amendment_sha256') != freeze['control_execution_amendment_sha256']):
                    raise ValueError('Only the hash-bound exact replay exception is allowed')
                replayed += 1
        result[key] = row; paths.add(path)
    if set(result) != wanted or paths != set(files) or historical != 71 or replayed != 1:
        raise ValueError('No missing, duplicate or extra final heads allowed')
    return result


def local_admission():
    home = manager.PUBLIC
    freeze = json.loads((home/'training_freeze.json').read_text())
    reg = json.loads(manager.REG.read_text())
    cfg = json.loads(manager.CONFIG.read_text())
    amendment = json.loads((home/'control_execution_amendment.json').read_text())
    if manager.base.sha(manager.CONFIG) != reg['config_sha256']:
        raise ValueError('Changed registered configuration')
    for rel,digest in {**reg['bindings'],**amendment['bindings']}.items():
        if manager.base.sha(ROOT/rel) != digest:
            raise ValueError('Changed registered code or protocol')
    if freeze['control_execution_amendment_sha256'] != manager.base.sha(home/'control_execution_amendment.json'):
        raise ValueError('Changed explicit execution amendment')
    files = {}
    for ref in freeze['fits']:
        path = (ROOT/ref['path']).resolve()
        if not path.is_relative_to(home.resolve()/'fits'):
            raise ValueError('Misplaced training metadata')
        files[ref['path']] = path.read_bytes()
    documents = validate(freeze,files,reg['expected'],cfg,manager.base.sha(manager.REG),amendment)
    collection = json.loads((home/'train_collection.json').read_text())
    if collection['artifact'] != freeze:
        raise ValueError('Collected freeze differs from current metadata')
    job = collection['join']['submission']['job_id']
    if job+'|COMPLETED|' not in collection['join']['accounting']['stdout']:
        raise ValueError('Successful complete-grid join required')
    inventory = dict(training_freeze_sha256=manager.base.sha(home/'training_freeze.json'),
        checkpoints={row['checkpoint']['path']:row['checkpoint'] for row in documents.values()})
    return documents,inventory


SERVER = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);home=sys.argv[2];heads=sys.argv[3];freeze_hash=sys.argv[4]
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1');root=root.resolve()
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
raw=(root/home/'training_freeze.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==freeze_hash
freeze=json.loads(raw);assert freeze['neural_fits']==144 and not freeze['validation_scored'];allowed={}
for ref in freeze['fits']:
 p=(root/ref['path']).resolve();assert p.is_relative_to(root/home/'fits')
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];cp=json.loads(raw)['checkpoint']
 p=(root/cp['path']).resolve();assert p.is_relative_to(root/heads) and cp['path'] not in allowed;allowed[cp['path']]=cp
assert len(allowed)==144
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin.buffer:
 assert len(line)<=4096;ref=json.loads(line);assert ref==allowed[ref['path']] and 0<ref['bytes']<=2*2**20
 raw=(root/ref['path']).read_bytes();assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
 print(json.dumps(ref),flush=True);sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
'''


class Reader(transport.NeuralReader):
    def __enter__(self):
        cmd = transport.with_keepalive(manager.base.ssh_args())+[shlex.join(['/usr/bin/python3','-c',SERVER,
            manager.REMOTE,HOME,HEADS,self.inventory['training_freeze_sha256']])]
        self.p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        try:
            if self.line() != {'ready':True}: raise ValueError('Owned complete final-head stream required')
        except BaseException:
            self.__exit__(*sys.exc_info()); raise
        return self
