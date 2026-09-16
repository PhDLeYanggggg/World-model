"""Audit local external raw files without converting or assigning experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_external_source_audit import SPECS, audit_source, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'external_data/OpenTraj/datasets')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/external_source_audit')
    args = parser.parse_args()
    if args.report_dir.resolve().is_relative_to(args.data_root.resolve()):
        raise SystemExit('Report output must not be inside the raw source tree')
    sources = []
    for source in SPECS:
        print(f'Auditing {source}', flush=True)
        item = audit_source(args.data_root, source)
        sources.append(item)
        print(f"{source}: {item['tracks_parsed']} tracks; {item['unique_agent_frame_points']} points; {item['parse_or_identity_failures']} issues", flush=True)
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'result_source': 'fresh_run',
        'scope': 'read_only_raw_source_availability_not_forecasting',
        'data_root': str(args.data_root), 'sources': sources,
        'no_interpolation': True, 'metric_or_seconds_promoted': False,
        'split_assigned': False, 'new_training': False, 'new_test_results': False,
        'stage5c_executed': False, 'smc_enabled': False,
        'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                        ('src/evaluation/m3w_external_source_audit.py', 'scripts/audit_m3w_external_sources.py')},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'availability.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.report_dir / 'availability.md').write_text(render_report(result))


if __name__ == '__main__':
    main()
