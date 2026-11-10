"""Verify nested completion envelopes and replay component resampling."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'protocol.md', 'verify.py', 'completion.py', 'shared.json', 'prior_evidence.zip'}
ORDER = ('original_lower', 'shared_lower', 'witness_low', 'witness_high', 'shared_upper', 'original_upper')


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected shared-seed replay inputs')
    for k, v in manifest.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed replay member')
    data = json.loads((root/'shared.json').read_text())
    draws = data['draws']
    if (len(draws) != 3000 or data['flights'] != 160 or data['seeds'] != [17, 29, 43]
            or set(data['points']) != {'independent', 'additive', 'same_family'}
            or any(len(v) != 14 or any(type(i) is not int or not 0 <= i < 14 for i in v) for v in draws)):
        raise ValueError('Unexpected shared-seed source design')
    estimate = runpy.run_path(str(root/'completion.py'))['estimate']
    summary = {}
    for a, r in data['points'].items():
        if set(r) != {'observed', *ORDER} or any(len(v) != 14 for v in r.values()):
            raise ValueError('Incomplete component means')
        for i, observed in enumerate(r['observed']):
            v = [r[k][i] for k in ORDER]
            if (any(x > y+1e-12 for x, y in zip(v, v[1:]))
                    or not isclose(v[0]+v[-1], 2*observed, abs_tol=1e-12)
                    or not isclose(v[1]+v[-2], 2*observed, abs_tol=1e-12)):
                raise ValueError('Witnesses must lie in nested symmetric outer bounds')
        summary[a] = {k: estimate(x, draws) for k, x in r.items()}
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, songdo_shared_seed=summary)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
