"""Fetch a pinned, code-only author snapshot; never download data or weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = json.loads((ROOT / 'configs/m3w_eqmotion_source.json').read_text())
    destination = ROOT / spec['destination']
    entries = {}
    for relative, expected in spec['files'].items():
        path = destination / relative
        source = 'cached_verified' if path.exists() else 'fresh_run'
        if path.exists():
            content = path.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/MediaBrain-SJTU/EqMotion/{spec["commit"]}/{relative}'
            content = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                                      '--max-time', '90', url], check=True, capture_output=True).stdout
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if len(content) != expected['size'] or blob != expected['git_blob_sha1']:
            raise ValueError(f'Author snapshot mismatch: {relative}; existing files not overwritten')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(content)
        entries[relative] = {**expected, 'sha256': hashlib.sha256(content).hexdigest(), 'result_source': source}
    report = {'repository': spec['repository'], 'commit': spec['commit'], 'license': spec['license'],
              'files': entries, 'total_bytes': sum(e['size'] for e in entries.values()),
              'datasets_downloaded': False, 'pretrained_weights_downloaded': False,
              'upstream_training_entry_executed': False}
    directory = ROOT / 'outputs/publication_readiness_2026_09/public_baselines'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'eqmotion_source_identity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'commit': spec['commit'], 'files_verified': len(entries), 'bytes': report['total_bytes']}))


if __name__ == '__main__':
    main()
