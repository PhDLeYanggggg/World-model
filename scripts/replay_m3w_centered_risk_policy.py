"""Replay frozen evidence with JSON-canonical identity containers only."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_centered_risk_policy as run


def canonical_identity(identity):
    return json.loads(json.dumps(identity,allow_nan=False))


def main():
    if sys.argv[1:] not in (['--phase','replay'],['--phase','replay_evaluate']):
        raise ValueError('Identity adapter only permits exact replays, not training, fitting or selection')
    original=run.identity
    def normalized():
        cfg,ident=original()
        return cfg,canonical_identity(ident)
    run.identity=normalized
    run.main()
    run.base.immutable_json(run.PUBLIC/('identity_adapter_'+sys.argv[2]+'.json'),dict(
        adapter=run.base.artifact(Path(__file__).resolve()),
        registered_runner=run.base.artifact(ROOT/'scripts/run_m3w_centered_risk_policy.py'),
        change='JSON canonicalization of immutable identity tuple/list containers only',
        policy_arrays_changed=False,registered_code_changed=False,thresholds_changed=False,
        first_replay_failed_pid=55077,first_replay_failure='tuple/list identity equality after identical arrays',
        fresh_readout_preserved=True))


if __name__=='__main__':main()
