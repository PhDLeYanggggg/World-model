"""Verify movement balance and replay anonymous location effects."""

from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'protocol.md', 'prior_evidence.zip', 'budget.json'}
ARMS = {'original', 'supported_merit', 'support_only'}


def replay(data):
    draws = data['draws']
    if (data['directions'] != 126 or data['locations'] != 7 or data['seeds'] != [17, 29, 43]
            or len(draws) != 3000 or any(len(v) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in v) for v in draws)
            or set(data['summary']) != {'original_quota', 'movement_matched'}):
        raise ValueError('Unexpected OpenDD or bootstrap roster')
    out = {}
    for mode, row in data['summary'].items():
        counts, contrasts, limits = (row[k] for k in ('counts', 'contrasts', 'movement_ratio_range'))
        if any(set(x) != ARMS for x in (counts, contrasts, limits)) or len({x['selected'] for x in counts.values()}) != 1:
            raise ValueError('Unexpected arms or intervention counts')
        derived = {}
        for arm, metrics in contrasts.items():
            c = counts[arm]
            if not 0 <= c['dual'] <= c['risk_pass'] <= c['selected']-c['undefined_easy']:
                raise ValueError('Unsupported easy risk cannot pass')
            for a, b in limits[arm].values():
                if not 0. < a <= b or (mode == 'movement_matched' and not .98999999 <= a <= b <= 1.01000001):
                    raise ValueError('Movement ratio outside its declared budget')
            available, lower, debit = (metrics[k]['location_values'] for k in
                                       ('ade_available_pp', 'ade_lower_pp', 'missing_debit_pp'))
            if any(z < 0. or not isclose(x-y, z, abs_tol=1e-12) for x, y, z in zip(available, lower, debit, strict=True)):
                raise ValueError('Invalid direct missing-label debit')
            derived[arm] = {}
            for name, row in metrics.items():
                x = row['location_values']
                if len(x) != 7 or not all(isfinite(i) for i in x):
                    raise ValueError('Incomplete location effects')
                vals = sorted(fsum(x[i] for i in v)/7 for v in draws)

                def q(p):
                    u = (len(vals)-1)*p
                    j = int(u)
                    return vals[j]+(u-j)*(vals[j+1]-vals[j])

                derived[arm][name] = dict(point=fsum(x)/7, ci95=[q(.025), q(.975)])
        out[mode] = dict(counts=counts, contrasts=derived, movement_ratio_range=limits)
    return out


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for k, v in hashes.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed replay input')
    result = replay(json.loads((root/'budget.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, opendd_fixed_joint_budget=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
