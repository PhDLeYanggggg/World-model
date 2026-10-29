"""Repeat exact readout checks without replacing first-run execution receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import run_m3w_reference_coverage as run
from scripts import report_m3w_reference_coverage as report
from scripts import analyze_m3w_reference_coverage_support as support
from src.evaluation.m3w_verification_receipt import retain_verified_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--readout', action='store_true')
    group.add_argument('--support', action='store_true')
    args = parser.parse_args()
    rules = {
        run.PUBLIC/'verification.json':(('exact_analysis_replay', 'analysis_sha256', 'record_csv_sha256'), 'exact_analysis_replay'),
        run.PUBLIC/'support_verification.json':(('exact_replay', 'summary_sha256'), 'exact_replay'),
    }
    original = run.immutable_json

    def writer(path, value):
        # Only runtime receipts are special. Every scientific artifact stays immutable.
        if Path(path) in rules:
            fields, flag = rules[Path(path)]
            return retain_verified_receipt(path, value, fields=fields, flag=flag, immutable_write=original)
        return original(path, value)

    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        receipt = run.PUBLIC/('verification.json' if args.readout else 'support_verification.json')
        before = run.digest(receipt) if receipt.exists() else None
        run.immutable_json = writer
        try:
            if args.readout:
                cfg, pc, data, jobs, identity = run.load()
                report.evaluate(cfg, pc, data, jobs, identity, True)
            else:
                support.run_diagnostic(True)
        finally:
            run.immutable_json = original
        after = run.digest(receipt)
        if before is not None and after != before:
            raise ValueError('Original execution receipt must remain unchanged')
        print(json.dumps(dict(mode='readout' if args.readout else 'support', exact_replay=True,
            original_execution_receipt_preserved=before is not None, receipt_sha256=after)))


if __name__ == '__main__':
    main()
