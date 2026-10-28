"""Preserve the owned /users alias while retaining canonical containment checks."""
import argparse
import json
import os
from pathlib import Path

from scripts.train_m3w_task_risk import NAME, PARENT, PUBLIC, PRIVATE, sha, once, quota, train


def verify(root):
    root = root.absolute(); source = root.parent/PARENT
    if root != Path('/users/k24101830/m3w')/NAME:
        raise ValueError('Owned experiment only')
    reg = json.loads((root/'registration.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    if json.loads((root/'.owner.json').read_text()) != dict(experiment=NAME, registration_sha256=sha(root/'registration.json')):
        raise ValueError('Wrong owner')
    for rel, digest in reg['bindings'].items():
        p = (root/'code'/rel).resolve()
        if not p.is_relative_to((root/'code').resolve()) or sha(p) != digest:
            raise ValueError('Changed bound source: '+rel)
    if sha(root/'config.json') != reg['config_sha256']:
        raise ValueError('Changed experiment settings')
    if (sha(source/'train_input_manifest.json') != reg['input_manifest_sha256']
            or json.loads((source/'.owner.json').read_text())['registration_sha256'] != reg['parent_registration_sha256']):
        raise ValueError('Changed source ownership or TRAIN manifest')
    manifest = json.loads((source/'train_input_manifest.json').read_text())
    if (manifest['input_role'] != 'source_TRAIN_only' or manifest['validation_rows_transferred']
            or manifest['independent_roles_read'] or len(manifest['heads']) != 72
            or cfg['head_training'] != json.loads((source/'config.json').read_text())['head_training']):
        raise ValueError('Frozen TRAIN-only scope required')
    return root, source, cfg, reg, manifest


def check_port(root):
    amendment = json.loads((root/'path_amendment.json').read_text())
    if amendment['training_registration_sha256'] != sha(root/'registration.json'):
        raise ValueError('Changed path amendment provenance')
    for rel, digest in amendment['bindings'].items():
        p = (root/'code'/rel).resolve()
        if not p.is_relative_to((root/'code').resolve()) or sha(p) != digest:
            raise ValueError('Changed portable execution binding')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['pilot','train'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    check_port(a.root)
    train(*verify(a.root), a.phase, a.resume)


if __name__ == '__main__':
    main()
