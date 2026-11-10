"""Replay the crossed path contrasts and check the original diagonal."""

from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'crossed.json', 'protocol.md', 'prior_evidence.zip'}
CONTRASTS = ('path_on_independent', 'path_on_joint', 'recipient_with_independent', 'recipient_with_joint', 'diagonal')


def replay(panel, original):
    draws, data = panel['draws'], panel['source_means']
    if (len(draws) != 3000 or set(data) != set(CONTRASTS)
            or any(len(v) != 14 or any(type(i) is not int or not 0 <= i < 14 for i in v) for v in draws)):
        raise ValueError('Invalid grouped replay inputs')
    result = {}
    for key, row in data.items():
        result[key] = {}
        for k, v in row.items():
            if v is None:
                result[key][k] = None
                continue
            if len(v) != 14 or not all(isfinite(x) for x in v):
                raise ValueError('Invalid component means')
            cut = quantiles([fsum(v[i] for i in ids)/14 for ids in draws], n=40, method='inclusive')
            result[key][k] = dict(point=fsum(v)/14, ci95=[cut[0], cut[-1]])
    for a, b in (('path_on_independent', 'recipient_with_joint'), ('recipient_with_independent', 'path_on_joint')):
        if not all(isclose(t, x+y, abs_tol=1e-12, rel_tol=0.) for t, x, y in zip(
                data['diagonal']['known_pp'], data[a]['known_pp'], data[b]['known_pp'], strict=True)):
            raise ValueError('Observed crossed differences do not telescope')
    for k, a in (('known_pp', 'observed_pp'), ('lower_pp', 'lower_pp'), ('upper_pp', 'upper_pp')):
        v, w = result['diagonal'][k], original[a]
        if not all(isclose(x, y, abs_tol=1e-12, rel_tol=0.) for x, y in zip(
                [v['point'], *v['ci95']], [w['point'], *w['ci95']], strict=True)):
            raise ValueError('The original diagonal changed')
    return result


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay members')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed replay member')
    with TemporaryDirectory() as tmp:
        sub = Path(tmp)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if (len(set(names)) != len(names)
                    or any(Path(n).name != n or n in ('', '.', '..') for n in names)):
                raise ValueError('Invalid inherited filename')
            for n in names:
                (sub/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    original = inherited['songdo_recipients']['independent']['total']
    return dict(inherited=inherited, fixed_recipient_paths=replay(json.loads((root/'crossed.json').read_text()), original))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
