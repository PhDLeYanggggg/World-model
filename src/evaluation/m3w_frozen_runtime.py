"""Execute exact historical model bytes without overwriting a newer checkout."""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys

from src.evaluation.m3w_experiment_contract import ExperimentContract,file_digest

SUPERVISED = 'src/world_model/m3w_supervised_intervention.py'
MIRROR_FILES = (SUPERVISED,'src/world_model/m3w_neural_gain_harm.py','src/world_model/m3w_oof_identity.py',
                'src/evaluation/m3w_development_evaluation.py','scripts/train_m3w_oof_cost_head.py')
MODULE_ORDER = (SUPERVISED,'src/world_model/m3w_neural_gain_harm.py','src/evaluation/m3w_development_evaluation.py')


def prepare_runtime(root, protocol, directory, *, revision, install):
    root,directory = Path(root).resolve(),Path(directory).resolve()
    if not directory.is_relative_to(root):raise ValueError('Frozen runtime must remain workspace-local')
    digests = {}
    for name in MIRROR_FILES:
        if name == SUPERVISED:
            value = subprocess.run(['git','show',revision+':'+name],cwd=root,check=True,capture_output=True).stdout
        else:
            value = (root/name).read_bytes()
        actual = hashlib.sha256(value).hexdigest()
        if name in protocol['bindings'] and actual != protocol['bindings'][name]:
            raise ValueError('Recovered code does not match original protocol: '+name)
        destination = directory/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != value:raise ValueError('Frozen code snapshot changed')
        else:
            destination.write_bytes(value)
        if file_digest(destination) != actual:raise ValueError('Frozen mirror write verification failed')
        digests[name] = actual
    if install:
        # Isolated runner process only. All dependent source-identity checks see
        # the exact recovered supervised bytes rather than a relaxed hash rule.
        for name in MODULE_ORDER:
            module_name = name[:-3].replace('/','.')
            if module_name in sys.modules:raise RuntimeError('Install frozen modules before current model imports')
            spec = importlib.util.spec_from_file_location(module_name,directory/name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try: spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name,None)
                raise
    return {'revision':revision,'files':digests,'relocation':{SUPERVISED:str((directory/SUPERVISED).relative_to(root))},
            'original_checkout_modified':False,'hash_checks_relaxed':False}


class RelocatedCodeContract(ExperimentContract):
    def __init__(self, protocol, root, artifacts, *, runtime):
        self._code_relocations = dict(runtime['relocation'])
        if set(self._code_relocations) != {SUPERVISED}:
            raise ValueError('Only the explicitly frozen historical code binding may be relocated')
        super().__init__(protocol,root,artifacts)

    def _path(self, relative):
        return super()._path(self._code_relocations.get(str(relative),relative))
