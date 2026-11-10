"""Replay fixed-proposal release contrasts from anonymous location means."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'protocol.md', 'prior_evidence.zip', 'release.json'}
ARMS = ('original', 'supported_merit', 'support_only')


def replay(data):
    draws = data['draws']
    if (len(draws) != 3000 or data['directions'] != 126 or data['seeds'] != [17, 29, 43]
            or any(len(x) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in x) for x in draws)):
        raise ValueError('Unexpected resampling design')
    result = {}
    for method, panel in data['summary'].items():
        counts = panel['counts']
        if set(counts) != set(ARMS) or method not in ('fixed_quarter', 'calibrated'):
            raise ValueError('Unexpected matched or release arms')
        for c in counts.values():
            if not (0 <= c['dual'] <= c['risk_pass'] <= c['selected']-c['undefined_easy']):
                raise ValueError('Undefined risk cannot pass')
        if method == 'fixed_quarter' and len({x['selected'] for x in counts.values()}) != 1:
            raise ValueError('Fixed-count comparison has unequal budgets')
        contrasts = {}
        for arm, row in panel['contrasts'].items():
            contrasts[arm] = {}
            for metric, v in row.items():
                x = v['location_values']
                if len(x) != 7 or not all(isfinite(i) for i in x):
                    raise ValueError('Missing paired location effect')
                draws_means = sorted(fsum(x[i] for i in d)/7 for d in draws)

                def q(p):
                    pos = (len(draws_means)-1)*p
                    i = int(pos)
                    return draws_means[i]+(pos-i)*(draws_means[i+1]-draws_means[i])

                contrasts[arm][metric] = dict(point=fsum(x)/7, ci95=[q(.025), q(.975)])
        result[method] = dict(counts=counts, contrasts=contrasts)
    if set(result) != {'fixed_quarter', 'calibrated'}:
        raise ValueError('Missing operating point')
    return result


def verify(root):
    root = Path(root)
    h = json.loads((root/'manifest.json').read_text())
    if set(h) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for k, v in h.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed replay input')
    result = replay(json.loads((root/'release.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, opendd_fixed_joint_release=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
