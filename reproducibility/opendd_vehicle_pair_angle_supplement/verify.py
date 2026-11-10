"""Reaggregate anonymised angle-target evidence, then replay its predecessor."""

from hashlib import sha256
from io import BytesIO
import json
from math import isclose, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'pair_angle.json', 'prior_evidence.zip'}


def replay(d):
    groups, draws = d['locations'], d['draws']
    if len(groups) != 7 or len(draws) != 3000 or any(len(x) != 7 or any(
            type(i) is not int or not 0 <= i < 7 for i in x) for x in draws):
        raise ValueError('Invalid bootstrap')
    counts, effects = groups[0]['counts'].keys(), groups[0]['effects'].keys()
    for row in groups:
        c, e = row['counts'], row['effects']
        if (c.keys() != counts or e.keys() != effects or any(type(x) is not int or x < 0 for x in c.values())
                or not all(isfinite(x) for x in e.values())):
            raise ValueError('Invalid site summaries')
    totals = {k: sum(r['counts'][k] for r in groups) for k in counts}
    for q, count in (('ego_count', 526), ('fixed_15pct', 1368)):
        for tag in ('pointwise', 'anchor_difference', 'direct'):
            for a in ('trajectory', 'agreement', 'calibrated_agreement'):
                prefix = f'{tag}.{q}.{a}'
                if (totals[f'{prefix}.rotated'] != count or not totals[f'{prefix}.qualified_joint_gain']
                        <= totals[f'{prefix}.swaps'] <= count):
                    raise ValueError('Changed intervention count')
    training = d['training']
    if training['rows'] != 44370 or any(type(x) is not int or not 0 <= x <= training['rows']
                                         for x in training.values()):
        raise ValueError('Changed matched training rows')
    intervals = {}
    for k in effects:
        values = [r['effects'][k] for r in groups]
        cuts = quantiles([sum(values[i] for i in x)/7 for x in draws], n=40, method='inclusive')
        intervals[k] = dict(point=sum(values)/7, ci95=[cuts[0], cuts[-1]])
    bounds = d['missing_label_bounds']
    for q in ('ego_count', 'fixed_15pct'):
        for tag in ('anchor_difference', 'pointwise', 'incumbent'):
            b = bounds[q][tag]
            key = (f'incumbent.direct.{q}.calibrated_agreement.relative' if tag == 'incumbent'
                   else f'direct_minus.{tag}.{q}.agreement.relative')
            lower, upper, obs, debit = (b[k]['point'] for k in ('lower', 'upper', 'observed', 'missing_debit'))
            if (debit < 0. or not isclose(lower, obs-debit, abs_tol=1e-12)
                    or not isclose(upper, obs+debit, abs_tol=1e-12)
                    or not isclose(lower, intervals[key]['point'], abs_tol=1e-12)):
                raise ValueError('Invalid missing-label bound')
    return dict(counts=totals, effects=intervals, training=training, missing_label_bounds=bounds)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed evidence file')
    with TemporaryDirectory() as folder:
        sub = Path(folder)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Invalid inherited filename')
            for name in names:
                (sub/name).write_bytes(z.read(name))
        old = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=old, pair_angle=replay(json.loads((root/'pair_angle.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
