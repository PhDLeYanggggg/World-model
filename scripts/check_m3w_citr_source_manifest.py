"""Compare local CITR raw object files with the author's public Git tree.

Only GitHub commit/tree metadata is downloaded. No data payload, authentication
secret, training, split assignment or license approval is part of this check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPO = 'dongfang-steven-yang/vci-dataset-citr'
PREFIX = 'data/trajectories/'


def github_json(path):
    # Use macOS's certificate trust; do not bypass TLS verification.
    response = subprocess.run(['/usr/bin/curl', '--fail', '--silent', '--show-error',
                               '--max-time', '30', '-H', 'Accept: application/vnd.github+json',
                               '-H', 'User-Agent: M3W-source-identity-audit',
                               'https://api.github.com/repos/' + REPO + path],
                              check=True, capture_output=True, text=True, timeout=35)
    return json.loads(response.stdout)


def git_blob_hash(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/external_source_audit/citr_upstream_identity.json')
    args = parser.parse_args()
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'result_source': 'fresh_run',
              'repository': 'https://github.com/' + REPO, 'raw_data_downloaded': False,
              'license_approved': False, 'independent_holdout_assigned': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        commit = github_json('/commits/master')
        tree = github_json('/git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')
        if tree.get('truncated'):
            raise ValueError('Incomplete remote tree')
        remote = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'
                  and e['path'].startswith(PREFIX) and e['path'].endswith('.csv')}
        root = ROOT / 'external_data/OpenTraj/datasets/CITR'
        local = {str(p.relative_to(root)): git_blob_hash(p) for p in (root / PREFIX).rglob('*.csv')}
        missing, extra = sorted(remote.keys() - local.keys()), sorted(local.keys() - remote.keys())
        mismatch = sorted(p for p in remote.keys() & local.keys() if remote[p] != local[p])
        result.update({
            'status': 'verified' if local and not (missing or extra or mismatch) else 'source_mismatch',
            'upstream_commit': commit['sha'], 'upstream_tree': commit['commit']['tree']['sha'],
            'raw_remote_object_files': len(remote), 'raw_local_object_files': len(local),
            'remote_pedestrian_files': sum(Path(p).name.startswith('p') for p in remote),
            'remote_vehicle_files': sum(Path(p).name.startswith('v') for p in remote),
            'missing_local_count': len(missing), 'extra_local_count': len(extra),
            'content_mismatch_count': len(mismatch),
            'missing_local_examples': missing[:10], 'extra_local_examples': extra[:10],
            'content_mismatch_examples': mismatch[:10],
            'raw_path_and_git_blob_manifest_sha256': hashlib.sha256(json.dumps(sorted(remote.items())).encode()).hexdigest(),
        })
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result.update(status='not_run_source_check_failed', reason=f'{type(exc).__name__}: {exc}')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
