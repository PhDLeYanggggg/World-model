"""Convert the existing pinned CITR raw files for diagnostic causal access only."""
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
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/OpenTraj/datasets/CITR')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/stage_cvpr2027_causal/citr_diagnostic')
    parser.add_argument('--upstream-report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/external_source_audit/citr_upstream_identity_system_tls.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/citr_causal_intake/build_report.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT) or not args.report.resolve().is_relative_to(ROOT):
        raise SystemExit('Derived output/report must remain inside the workspace')
    from src.data_unification.m3w_citr_recordings import build_citr, check_past_only
    began = time.monotonic()

    def progress(name, done, total):
        with (args.output_dir / 'heartbeat.jsonl').open('a') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'recording': name, 'completed': done, 'total': total,
                                     'elapsed_seconds': time.monotonic()-began})+'\n')
        print(json.dumps({'recording': name, 'completed': done, 'total': total}), flush=True)

    result = build_citr(args.source_root, args.output_dir, json.loads(args.upstream_report.read_text()),
                        resume=args.resume, progress=progress)
    result['past_only_check'] = check_past_only(args.output_dir, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, args.report)
    print(json.dumps({k: result[k] for k in ('result_source', 'recording_count', 'points', 'agents', 'raw_exact_windows',
                                           'physical_scene_groups', 'training_run')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
