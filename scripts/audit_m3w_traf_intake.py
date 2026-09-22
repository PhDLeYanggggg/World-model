"""Inventory TRAF raw annotation support without assigning scientific data roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_traf_intake import audit_directory, render_report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/OpenTraj')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs/publication_readiness_2026_09/traf_intake_v1')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(source) or not output.is_relative_to(ROOT):
        raise SystemExit('Output must be inside workspace and outside raw source tree')
    directory = source / 'datasets/TRAF/Annotated Ground Truth Files'
    print('Auditing raw frame/ID/box structure; no geometry conversion or forecasts', flush=True)
    result = audit_directory(directory)
    result['source_checkout_commit'] = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    result['local_source_sha256'] = {str(p.relative_to(source)): digest(p) for p in (
        source / 'README.md', source / 'datasets/TRAF/README.md', source / 'LICENSE.txt')}
    result['implementation_sha256'] = {name: digest(ROOT / name) for name in (
        'src/evaluation/m3w_traf_intake.py', 'src/evaluation/m3w_external_source_audit.py',
        'scripts/audit_m3w_traf_intake.py', 'tests/test_m3w_traf_intake.py')}
    path = output / 'analysis.json'
    # JSON round-trip also checks that every output is finite and portable.
    result = json.loads(json.dumps(result, allow_nan=False))
    if args.verify:
        if result != json.loads(path.read_text()):
            raise SystemExit('Fresh raw recount differs from stored intake audit')
        print('Exact recount verified; result_source=cached_verified', flush=True)
    else:
        if path.exists():
            raise SystemExit('Audit exists; use --verify or a new versioned output directory')
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (output / 'availability.md').write_text(render_report(result))
    print(json.dumps({'files': result['annotation_file_count'],
                      'classes': result['class_counts_before_quarantine'],
                      'quarantined': result['quarantined_recordings'],
                      'bbox_checks': result['bbox_checks'],
                      'analysis_sha256': digest(path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
