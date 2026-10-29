"""Replay frozen record-policy readout through an inference-only shape adapter."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_record_policy as run
from src.evaluation.m3w_multiaction_record_inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    paths = (Path(__file__), ROOT/'src/evaluation/m3w_multiaction_record_inference.py',
             ROOT/'tests/test_m3w_multiaction_record_inference.py', run.PUBLIC/'inference_amendment.md')
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in paths:
            run.parent.committed(path)
        run.once(run.PUBLIC/'inference_amendment.json', dict(
            bindings={str(p.relative_to(ROOT)):run.sha(p) for p in paths},
            training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
            frozen_training_api_sha256=run.sha(ROOT/'src/world_model/m3w_multiaction_record_policy.py'),
            first_failed_readout_pid=60443, first_failed_readout_exit_code=1,
            cause='singleton_Torch_index_collapses_NumPy_candidate_axis',
            weights_optimizer_and_training_unchanged=True, multi_action_parity_exact=True,
            singleton_tensor_reference_test_pass=True, failed_attempt_created_groups=False))
        old = run.api.predict; run.api.predict = predict
        try:
            run.evaluate(verify=args.verify, resume=args.resume)
        finally:
            run.api.predict = old


if __name__ == '__main__':
    main()
