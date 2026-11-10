"""Replay the paired location statistics for temporal label masking."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'equal_coverage.pdf',
           'pair_label_stress.json', 'prior_evidence.zip'}


def near(a, b):
    return isclose(a, b, rel_tol=0., abs_tol=1e-10)


def replay(data):
    draws, support, results = (data[k] for k in ('draws', 'support', 'results'))
    if (len(draws) != 3000 or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)
            or support['queries'] != 3711 or support['pair_steps'] < 3711*12
            or support['pair_steps'] % 12):
        raise ValueError('Unexpected complete query cohort')

    def values(stat):
        x = stat['locations']
        if len(x) != 7:
            raise ValueError('Require the seven location means')
        sample = sorted(mean(x[i] for i in d) for d in draws)

        def quantile(p):
            i, f = divmod(p*(len(sample)-1), 1)
            i = int(i)
            return sample[i] + f*(sample[i+1]-sample[i])

        if not all(near(a, b) for a, b in zip([mean(x), quantile(.025), quantile(.975)],
                                               [stat['point'], *stat['ci95']], strict=True)):
            raise ValueError('Location bootstrap arithmetic differs')
        return x

    if set(results) != {'eq_neighbors_minus_ego', 'social_neighbors_minus_ego', 'completion_minus_cv'}:
        raise ValueError('Original contrasts differ')
    for row in results.values():
        full = values(row['full'])
        if set(row['masked']) != {'3', '6'}:
            raise ValueError('Unexpected masking severity')
        for n, masked in row['masked'].items():
            if masked['retained_pair_steps'] != support['pair_steps']*(12-int(n))//12:
                raise ValueError('Masking coverage differs from the equal-count control')
            arms = {}
            for name, arm in masked['arms'].items():
                arms[name], bias, lo, hi = (values(arm[k]) for k in ('observed', 'bias', 'lower', 'upper'))
                if any(not near(b, p-q) or not l-1e-10 <= q <= h+1e-10
                       for p, b, q, l, h in zip(arms[name], bias, full, lo, hi, strict=True)):
                    raise ValueError('Complete truth or bias is inconsistent')
            diff = values(masked['first_minus_last'])
            if any(not near(a-b, c) for a, b, c in zip(arms['first'], arms['last'], diff, strict=True)):
                raise ValueError('First-minus-last contrast differs')
    return dict(support=support, results=results)


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected replay members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Replay member changed')
    current = replay(json.loads((root/'pair_label_stress.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        if (root/'supplement.pdf').read_bytes() != (folder/'supplement.pdf').read_bytes():
            raise ValueError('Reader PDF and inherited evidence differ')
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    if parent['opendd_backbone_relative']['support']['complete_pair_queries'] != current['support']['queries']:
        raise ValueError('Inherited complete cohort count differs')
    return dict(inherited=parent, opendd_pair_label_stress=current)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
