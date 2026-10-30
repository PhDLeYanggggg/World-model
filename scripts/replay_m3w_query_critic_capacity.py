"""Replay the frozen diagnostic after its parent's completion event has changed."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import probe_m3w_query_critic_capacity as frozen
from src.evaluation.m3w_capacity_creation_event import FIELD, preserve_creation_event

REG = frozen.PUBLIC/'execution_amendment.json'
PROTOCOL = frozen.PUBLIC/'execution_amendment.md'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_capacity_creation_event.py',
    ROOT/'tests/test_m3w_capacity_creation_event.py', PROTOCOL)


def admission(register=False):
    recorded = json.loads(frozen.REG.read_text())
    refusal = frozen.PRIVATE/'post_parent_complete_entrypoint.log'
    if ('Existing immutable evidence differs:' not in refusal.read_text()
            or type(recorded[FIELD]) is not bool
            or not (frozen.parent.PUBLIC/'complete.json').exists()):
        raise ValueError('Observed historical-event replay refusal required')
    frozen.once(REG, dict(bindings={str(p.relative_to(ROOT)): frozen.sha(p) for p in FILES},
        original_registration_sha256=frozen.sha(frozen.REG),
        original_complete_sha256=frozen.sha(frozen.PUBLIC/'complete.json'),
        original_summary_sha256=frozen.sha(frozen.PUBLIC/'summary.json'),
        original_verification_sha256=frozen.sha(frozen.PUBLIC/'verification.json'),
        observed_refusal_log_sha256=frozen.sha(refusal),
        original_entrypoint_exit_code=1, preserved_historical_field=FIELD,
        preserved_historical_value=recorded[FIELD],
        scientific_definition_and_all_original_results_unchanged=True,
        new_training_updates=0, new_policy_scored=False, independent_roles_opened=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if frozen.sha(ROOT/rel) != digest: raise ValueError('Execution amendment changed: '+rel)
            frozen.parent.frozen.BASE.engine.committed(ROOT/rel)
        frozen.parent.frozen.BASE.engine.committed(REG)


@contextmanager
def preserve_registration_event():
    original = frozen.once
    def checked(path, value):
        if Path(path).resolve() == frozen.REG.resolve():
            value = preserve_creation_event(value, json.loads(frozen.REG.read_text()))
        return original(path, value)
    frozen.once = checked
    try: yield
    finally: frozen.once = original


def verify():
    admission()
    start = time.monotonic()
    with preserve_registration_event(): frozen.execute(verify=True)
    check = json.loads((frozen.PUBLIC/'verification.json').read_text())
    if not check['all72groups_exact']: raise ValueError('Original complete replay must remain exact')
    frozen.once(frozen.PUBLIC/'execution_replay.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        all72groups_exact=True, original_registration_sha256=frozen.sha(frozen.REG),
        summary_sha256=frozen.sha(frozen.PUBLIC/'summary.json'),
        original_verification_sha256=frozen.sha(frozen.PUBLIC/'verification.json'),
        execution_amendment_sha256=frozen.sha(REG), original_results_overwritten=False,
        scientific_definition_changed=False, new_training_updates=0, independent_roles_opened=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'verify'))
    args = p.parse_args()
    with (frozen.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if args.phase == 'register' else verify()
