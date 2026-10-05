"""Hash-verified, cache-free checkpoint access for the frozen seven-arm readout."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

from scripts import run_m3w_temporal_auxiliary_readout as original
from scripts import manage_m3w_temporal_auxiliary_create as manager
from scripts.m3w_bounded_packet_write import with_keepalive, write_payload
from scripts.run_m3w_quality_components import Reader, read_bytes

ROOT, PUBLIC, PRIVATE = original.ROOT, original.PUBLIC, original.PRIVATE
HOME = str(original.training.PUBLIC.relative_to(ROOT))
HEADS = 'data/stage_cvpr2027_experiments/'+original.training.NAME+'/heads'
REGISTRATION = PUBLIC/'create_reader_registration_v5.json'
INVENTORY = original.training.PUBLIC/'create_training_inventory.json'
METADATA_CAP = 16*2**20
CHECKPOINT_CAP = 2*2**20


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_relative(value, prefix):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value or not p.is_relative_to(prefix):
        raise ValueError('Artifact outside the registered experiment')
    return p


def documents(freeze_text, files, expected, cfg, registration_sha256):
    """Validate the full metadata grid without opening tensors or scientific data."""
    freeze = json.loads(freeze_text)
    if (freeze['neural_fits'] != cfg['neural_fits'] or freeze['validation_labels_scored'] is not False
            or freeze['independent_roles_read'] is not False or freeze['deployment_changed'] is not False
            or freeze['new_forecasters'] is not False):
        raise ValueError('Complete, unscored, fixed-final training required')
    wanted = {(r['group'], r['source'], r['head_seed'], arm) for r in expected for arm in original.training.api.ARMS}
    found = {}; paths = set(); total = len(freeze_text.encode()); refs = set()
    for ref in freeze['fits']:
        rel = ref['path']; safe_relative(rel, HOME+'/fits')
        if rel in refs or rel not in files: raise ValueError('Missing or duplicate training receipt')
        refs.add(rel); raw = files[rel].encode(); total += len(raw)
        if digest(raw) != ref['sha256']: raise ValueError('Training receipt hash mismatch')
        row = json.loads(raw); identity = row['identity']; cp = row['checkpoint']
        key = identity['group'], identity['source'], identity['seed'], row['arm']
        if key not in wanted or key in found or row['step'] != cfg['head_training']['steps']:
            raise ValueError('Missing, duplicate or partial fixed-final head')
        if row['validation_labels_scored'] is not False or identity['registration_sha256'] != registration_sha256:
            raise ValueError('Scored or unregistered checkpoint')
        safe_relative(cp['path'], HEADS)
        name = identity['group']+'_head'+str(identity['seed'])+'_'+row['arm']
        if cp['path'] != HEADS+'/'+name+'/checkpoint.pt.gz' or cp['path'] in paths:
            raise ValueError('Checkpoint identity/path mismatch')
        if not 0 < cp['bytes'] <= CHECKPOINT_CAP or not re.fullmatch('[0-9a-f]{64}', cp['sha256']):
            raise ValueError('Invalid checkpoint cap/hash')
        paths.add(cp['path']); found[key] = row
    if set(found) != wanted or len(found) != cfg['neural_fits'] or refs != set(files):
        raise ValueError('Every registered head and no extra receipt required')
    if total > METADATA_CAP or sum(r['checkpoint']['bytes'] for r in found.values()) > cfg['checkpoint_cap_bytes']:
        raise ValueError('Registered metadata/checkpoint cap exceeded')
    return found


def check_bundle(bundle, expected, cfg, registration_sha256, manifest_sha256, job_id):
    if (bundle['job_id'] != job_id or bundle['accounting'] != 'COMPLETED|0:0'
            or bundle['manifest_sha256'] != manifest_sha256
            or bundle['registration_sha256'] != registration_sha256
            or bundle['code_hashes_verified'] is not True):
        raise ValueError('Successful owned training and exact manifest required')
    result = documents(bundle['freeze'], bundle['files'], expected, cfg, registration_sha256)
    expected_cp = {r['checkpoint']['path']: r['checkpoint'] for r in result.values()}
    if bundle['checkpoints'] != expected_cp:
        raise ValueError('Every remote checkpoint must be checksum verified')
    return result


COLLECT = r'''
import base64,gzip,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);job=sys.argv[2];home=sys.argv[3];heads=sys.argv[4];cap=int(sys.argv[5])
assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve()
owner=json.loads((root/'.owner.json').read_text());assert owner['experiment']==root.name
assert json.loads((root/'train_submission.json').read_text())['job_id']==job and job.isdigit()
q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
manifest_raw=(root/'train_input_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
assert hashlib.sha256(manifest_raw).hexdigest()==sys.argv[6]
assert manifest['registration_sha256']==owner['registration_sha256']
assert hashlib.sha256((root/'config.json').read_bytes()).hexdigest()==manifest['config_sha256']
for rel,h in manifest['code_bindings'].items():
 p=(root/'code'/rel).resolve();assert p.is_relative_to(root/'code')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
freeze=(root/home/'training_freeze.json').read_text();files={};checks={};total=len(freeze.encode())
for ref in json.loads(freeze)['fits']:
 rel=pathlib.PurePosixPath(ref['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.is_relative_to(home+'/fits')
 p=(root/rel).resolve();assert p.is_relative_to(root/home/'fits') and p.stat().st_size<=cap
 raw=p.read_bytes();total+=len(raw);assert total<=cap and hashlib.sha256(raw).hexdigest()==ref['sha256']
 assert str(rel) not in files;files[str(rel)]=raw.decode();cp=json.loads(raw)['checkpoint'];crel=pathlib.PurePosixPath(cp['path'])
 assert not crel.is_absolute() and '..' not in crel.parts and crel.is_relative_to(heads)
 p=(root/crel).resolve();assert p.is_relative_to(root/heads) and p.stat().st_size==cp['bytes'] and 0<cp['bytes']<=2*2**20
 assert hashlib.sha256(p.read_bytes()).hexdigest()==cp['sha256'];assert cp['path'] not in checks;checks[cp['path']]=cp
out=dict(job_id=job,accounting=q.stdout.strip(),registration_sha256=owner['registration_sha256'],
 manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),code_hashes_verified=True,freeze=freeze,files=files,checkpoints=checks)
raw=json.dumps(out).encode();assert len(raw)<=2*cap
print(json.dumps(dict(payload=base64.b64encode(gzip.compress(raw)).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
'''


def write_exact(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw: raise ValueError('Preserve conflicting existing artifact: '+str(path))
    else:
        with path.open('xb') as f: f.write(raw)


def collect(cfg, reg):
    # This local receipt exists only after the full-training submission. A pilot
    # or stale heartbeat never authorizes checkpoint collection or readout.
    job = json.loads((manager.PRIVATE/'create_train_submission.json').read_text())['job_id']
    manifest_hash = manager.sha(manager.PRIVATE/'create_train_input_manifest.json')
    out = manager.remote(COLLECT, [manager.REMOTE, job, HOME, HEADS, str(METADATA_CAP), manifest_hash], timeout=90)
    if not 0 < out['bytes'] <= 2*METADATA_CAP: raise ValueError('Metadata bundle too large')
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(out['payload'], validate=True))) as f:
        raw = f.read(2*METADATA_CAP+1)
    if len(raw) != out['bytes'] or digest(raw) != out['sha256']: raise ValueError('Metadata bundle checksum mismatch')
    bundle = json.loads(raw)
    docs = check_bundle(bundle, reg['expected_source_heads'], cfg, original.sha(original.training.PUBLIC/'registration.json'), manifest_hash, job)
    for rel in bundle['files']:
        if not (ROOT/rel).resolve().is_relative_to(ROOT/ HOME/'fits'):
            raise ValueError('Local receipt path redirects outside the owned experiment')
    for rel, text in bundle['files'].items(): write_exact(ROOT/rel, text.encode())
    write_exact(original.training.PUBLIC/'training_freeze.json', bundle['freeze'].encode())
    inventory = dict(job_id=job, accounting=bundle['accounting'], metadata_bundle_sha256=out['sha256'],
        training_freeze_sha256=digest(bundle['freeze'].encode()), manifest_sha256=manifest_hash,
        checkpoints=bundle['checkpoints'], code_hashes_verified=True, checkpoint_payloads_saved_locally=False,
        validation_predictions_scored=False, independent_roles_read=False, result_source='cached_verified_CREATE_final_training')
    original.once(INVENTORY, inventory)
    print(json.dumps(dict(complete_training_heads=len(docs), job_id=job,
                         metadata_collected=True, checkpoints_saved_locally=False, validation_scored=False)))


SERVER = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);home=sys.argv[2];heads=sys.argv[3];freeze_hash=sys.argv[4]
assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve()
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
raw=(root/home/'training_freeze.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==freeze_hash
allowed={}
for ref in json.loads(raw)['fits']:
 p=(root/ref['path']).resolve();assert p.is_relative_to(root/home/'fits')
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];cp=json.loads(raw)['checkpoint']
 p=(root/cp['path']).resolve();assert p.is_relative_to(root/heads);allowed[cp['path']]=cp
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin.buffer:
 assert len(line)<=4096;ref=json.loads(line);assert ref==allowed[ref['path']] and 0<ref['bytes']<=2*2**20
 raw=(root/ref['path']).read_bytes();assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
 print(json.dumps(ref),flush=True);sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
'''


def decode_checkpoint(raw, ref):
    if not 0 < ref['bytes'] <= CHECKPOINT_CAP or len(raw) != ref['bytes'] or digest(raw) != ref['sha256']:
        raise ValueError('Checkpoint checksum/cap mismatch before deserialization')
    # Only a precommitted, owned, hash-verified checkpoint is deserialized.
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
        return original.training.api.torch.load(f, map_location='cpu', weights_only=False)


class NeuralReader(Reader):
    def __init__(self, inventory):
        self.inventory = inventory
        self.refs = {(ROOT/k).resolve(): v for k,v in inventory['checkpoints'].items()}
        self.fetched_bytes = 0

    def __enter__(self):
        command = with_keepalive(manager.ssh_args())+[shlex.join(['/usr/bin/python3', '-c', SERVER,
            manager.REMOTE, HOME, HEADS, self.inventory['training_freeze_sha256']])]
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        try:
            if self.line() != {'ready': True}: raise ValueError('Owned frozen training stream required')
        except BaseException:
            self.__exit__(*sys.exc_info()); raise
        return self

    def load(self, path):
        ref = self.refs[Path(path).resolve()]
        write_payload(self.p.stdin.fileno(), (json.dumps(ref)+'\n').encode())
        if self.line() != ref: raise ValueError('Remote checkpoint identity changed')
        raw = read_bytes(self.p.stdout, ref['bytes'])
        state = decode_checkpoint(raw, ref); self.fetched_bytes += len(raw)
        return state


@contextmanager
def bound_loader(load):
    previous = original.training.api.core.read_checkpoint
    original.training.api.core.read_checkpoint = load
    try: yield
    finally: original.training.api.core.read_checkpoint = previous


def local_admission(cfg, reg):
    home = original.training.PUBLIC
    if not (home/'training_freeze.json').exists() or not INVENTORY.exists():
        return None, None
    freeze = (home/'training_freeze.json').read_text()
    files = {r['path']: (ROOT/r['path']).read_text() for r in json.loads(freeze)['fits']
             if safe_relative(r['path'], HOME+'/fits')}
    result = documents(freeze, files, reg['expected_source_heads'], cfg, original.sha(home/'registration.json'))
    inventory = json.loads(INVENTORY.read_text())
    if (digest(freeze.encode()) != inventory['training_freeze_sha256']
            or inventory['checkpoints'] != {d['checkpoint']['path']: d['checkpoint'] for d in result.values()}
            or inventory['accounting'] != 'COMPLETED|0:0'
            or inventory['validation_predictions_scored'] is not False):
        raise ValueError('Changed training collection receipt')
    return result, inventory


def registration():
    cfg, reg = original.registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    paths = original.parent.forest.closure(ROOT, ['scripts.read_m3w_temporal_auxiliary_create'])
    paths += [PUBLIC/'create_reader_protocol.md', PUBLIC/'create_reader_path_fix.md',
              ROOT/'tests/test_m3w_temporal_create_reader.py']
    extension = dict(execution_revision=5,
        previous_reader_registration_sha256=original.sha(PUBLIC/'create_reader_registration_v4.json'),
        bindings={str(p.relative_to(ROOT)): original.sha(p) for p in paths},
        original_readout_registration_sha256=original.sha(PUBLIC/'registration.json'),
        original_training_registration_sha256=original.sha(original.training.PUBLIC/'registration.json'),
        transport_only=True, original_evaluator_unchanged=True, checkpoint_disk_cache=False,
        new_validation_predictions_scored=False, independent_roles_read=False, risk_budget=cfg['risk_budget'])
    return cfg, reg, extension


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'collect', 'preflight', 'run', 'verify'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg, reg, extension = registration()
    if args.phase == 'register': original.once(REGISTRATION, extension); return
    assert extension == json.loads(REGISTRATION.read_text())
    original.parent.base.inter.committed(REGISTRATION)
    if args.phase == 'collect': collect(cfg, reg); return
    docs, inventory = local_admission(cfg, reg)
    if args.phase == 'preflight':
        print(json.dumps(dict(readout_allowed=docs is not None, neural_fits_verified=0 if docs is None else len(docs),
            scientific_arrays_loaded=False, remote_contacted=False, validation_predictions_scored=False)))
        return
    if docs is None: raise RuntimeError('All final heads must finish, be collected and committed before readout')
    for path in (original.training.PUBLIC/'training_freeze.json', INVENTORY): original.parent.base.inter.committed(path)
    if args.phase == 'run': assert not (PUBLIC/'complete.json').exists()
    else: assert (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    original.training.api.torch.set_num_threads(cfg['cpu_threads']); original.training.api.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with NeuralReader(inventory) as reader, bound_loader(reader.load):
            original.run(cfg, reg, docs, resume=args.resume, verify=args.phase == 'verify')
        original.once(PUBLIC/('create_reader_verification.json' if args.phase == 'verify' else 'create_reader_complete.json'),
            dict(checkpoint_bytes_fetched=reader.fetched_bytes, training_freeze_sha256=inventory['training_freeze_sha256'],
                 reader_registration_sha256=original.sha(REGISTRATION), no_checkpoint_disk_cache=True,
                 original_evaluator_unchanged=True, new_training=False, independent_roles_read=False))


if __name__ == '__main__':
    main()
