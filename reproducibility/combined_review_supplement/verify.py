"""Verify the supplementary archive and replay both frozen evidence sets."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


REQUIRED = {'reading_guide.pdf', 'technical_appendix.pdf', 'verify.py',
            'reviewer_evidence.zip', 'moving_peak_certificate.zip'}


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if found != set(manifest) | {'manifest.json'} or not REQUIRED <= set(manifest):
        raise ValueError('Supplement archive roster differs from the manifest')
    for name, digest in manifest.items():
        if sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Supplement hash mismatch: {name}')

    def replay(name):
        with TemporaryDirectory() as tmp, ZipFile(root/name) as z:
            z.extractall(tmp)
            run = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=tmp,
                                 capture_output=True, text=True, check=True, timeout=90)
        return json.loads(run.stdout)

    stats, moving = replay('reviewer_evidence.zip'), replay('moving_peak_certificate.zip')
    expected = stats['headline']['moving']
    if (moving['views'], moving['recordings'], moving['upper_bounds']['record']['excluded']) != (
            expected['views'], expected['recordings'], expected['ols4_excluded']):
        raise ValueError('The independent dual certificate disagrees with the headline replay')
    moving.pop('per_view')
    return dict(**stats, moving_peak_certificate=moving, verified_members=len(manifest))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
