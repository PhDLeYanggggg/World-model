"""Recompute the observability and decision contrasts from location means."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'protocol.md', 'prior_evidence.zip', 'visibility.json'}


def replay(data):
    draws, row = data['draws'], data['summary']
    if (len(draws) != 3000 or data['directions'] != 126 or data['fits'] != 63
            or data['seeds'] != [17, 29, 43]
            or any(len(x) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in x) for x in draws)):
        raise ValueError('Unexpected fit or resampling design')
    if set(row['counts']) != {'support', 'profile', 'learned'} or any(
            c['selected'] != 2196 or not 0 <= c['max_spend_error'] <= .01000001
            or not 0 <= c['dual'] <= c['risk_pass'] <= c['selected']-c['undefined_easy']
            for c in row['counts'].values()):
        raise ValueError('Unmatched edits or unsupported risk pass')
    summary = {}
    for name, metrics in row['contrasts'].items():
        summary[name] = {}
        for k, m in metrics.items():
            x = m['location_values']
            if len(x) != 7 or not all(isfinite(i) for i in x):
                raise ValueError('Incomplete paired location values')
            values = sorted(fsum(x[i] for i in d)/7 for d in draws)

            def q(p):
                u = (len(values)-1)*p
                j = int(u)
                return values[j]+(u-j)*(values[j+1]-values[j])

            summary[name][k] = dict(point=fsum(x)/7, ci95=[q(.025), q(.975)])
    return dict(counts=row['counts'], contrasts=summary, mask_metrics=row['mask_metrics'])


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay inputs')
    for k, v in manifest.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed archive member')
    result = replay(json.loads((root/'visibility.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, opendd_step_visibility=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
