"""Reproduce the grouped recipient contrast and the original Songdo total."""

from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'attribution.json', 'protocol.md', 'prior_evidence.zip'}
CONTROLS = ('independent', 'additive', 'same_family')
PARTS = ('joint_only', 'control_only', 'shared')


def replay(panel, original):
    draws = panel['draws']
    if (len(draws) != 3000 or set(panel['groups']) != set(CONTROLS)
            or any(len(v) != 14 or any(type(i) is not int or not 0 <= i < 14 for i in v) for v in draws)):
        raise ValueError('Invalid paired source design')
    result = {}
    for a, g in panel['groups'].items():
        if set(g) != set(PARTS):
            raise ValueError('Recipient decomposition has missing parts')
        for s in g.values():
            o, d = (s[k] for k in ('observed_pp', 'missing_debit_pp'))
            if len(o) != 14 or len(d) != 14 or not all(isfinite(x) and isfinite(y) and y >= 0
                                                        for x, y in zip(o, d, strict=True)):
                raise ValueError('Invalid source-group contributions')
        effects = {}
        terms = dict(g)
        for key, names in (('allocation', PARTS[:2]), ('total', PARTS)):
            terms[key] = {m: [fsum(g[p][m][i] for p in names) for i in range(14)]
                          for m in ('observed_pp', 'missing_debit_pp')}
        for name, s in terms.items():
            o, d = s['observed_pp'], s['missing_debit_pp']
            metrics = dict(s, lower_pp=[x-y for x, y in zip(o, d, strict=True)],
                           upper_pp=[x+y for x, y in zip(o, d, strict=True)])
            effects[name] = {}
            for metric, v in metrics.items():
                q = quantiles([fsum(v[i] for i in row)/14 for row in draws], n=40, method='inclusive')
                effects[name][metric] = dict(point=fsum(v)/14, ci95=[q[0], q[-1]])
        for k in ('observed', 'lower', 'upper'):
            v = effects['total'][k+'_pp']
            base = original[a]
            if not all(isclose(x, y, abs_tol=1e-12, rel_tol=0.) for x, y in zip(
                    [v['point'], *v['ci95']], [base[k], *base[k+'_ci95']], strict=True)):
                raise ValueError('The primary Songdo result changed')
        result[a] = effects
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
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Invalid inherited filename')
            for n in names:
                (sub/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    original = inherited['inherited']['inherited']['external_completion']['songdo']
    return dict(inherited=inherited, songdo_recipients=replay(json.loads((root/'attribution.json').read_text()), original))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
