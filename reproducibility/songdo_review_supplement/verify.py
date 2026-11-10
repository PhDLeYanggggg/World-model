"""Verify the current review bundle and independently replay all summaries."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    expected = {'README.md', 'verify.py', 'songdo_confirmation.pdf', 'songdo_replay.zip', 'original_evidence.zip'}
    if set(manifest) != expected or {p.name for p in root.iterdir()} != expected | {'manifest.json'}:
        raise ValueError('Unexpected supplementary files')
    if any(sha256((root/k).read_bytes()).hexdigest() != v for k, v in manifest.items()):
        raise ValueError('Supplement hash mismatch')

    def replay(name):
        with TemporaryDirectory() as tmp, ZipFile(root/name) as z:
            z.extractall(tmp)
            result = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=tmp, check=True,
                                    capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)

    return dict(original=replay('original_evidence.zip'), songdo=replay('songdo_replay.zip'))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
