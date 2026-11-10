"""Reconstruct the OpenDD direct completion contrasts and the prior evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'protocol.md', 'verify.py', 'completion.py', 'points.json', 'prior_evidence.zip'}


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected review package members')
    for k, v in manifest.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed review package member')
    d = json.loads((root/'points.json').read_text())
    if (set(d['points']) != {'support_count', 'support_count_movement', 'learned_minus_profile'}
            or len(d['draws']) != 3000 or any(len(x['observed']) != 7 for x in d['points'].values())):
        raise ValueError('Wrong OpenDD roster')
    fn = runpy.run_path(str(root/'completion.py'))['completion']
    result = {a: fn(**v, draws=d['draws']) for a, v in d['points'].items()}
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(names) != len(set(names)) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        previous = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=previous, opendd_completion_signs=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
