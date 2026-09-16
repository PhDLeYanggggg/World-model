"""Build a causal DUT diagnostic cache from the pinned author raw annotation set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/DUT_author_raw')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/stage_cvpr2027_causal/dut_diagnostic')
    parser.add_argument('--upstream-report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/dut_causal_intake/source_manifest.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/dut_causal_intake/build_report.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / 'data') or not args.report.resolve().is_relative_to(ROOT / 'outputs'):
        raise SystemExit('Keep derived data and reports inside the workspace')
    from src.data_unification.m3w_dut_recordings import build_dut, check_past_only
    began = time.monotonic()

    def progress(name, done, total):
        record = {'pid': os.getpid(), 'recording': name, 'completed': done, 'total': total,
                  'elapsed_seconds': time.monotonic()-began}
        with (args.output_dir / 'heartbeat.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    result = build_dut(args.source_root, args.output_dir, json.loads(args.upstream_report.read_text()),
                       resume=args.resume, progress=progress)
    result['past_only_check'] = check_past_only(args.output_dir, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, args.report)
    print(json.dumps({k: v for k,v in result.items() if k not in ('recordings', 'past_only_check')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
