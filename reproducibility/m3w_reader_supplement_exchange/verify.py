"""Replay the anonymous same-mask exchange table and inherited evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'exchange.json', 'prior_evidence.zip'}
ARMS = ('incumbent', 'aligned', 'uniform', 'mask_oracle', 'menu_or_keep')
FIELDS = ('original', 'full_grid', 'material_full_grid')
CONTRASTS = (('menu_or_keep', 'incumbent'), ('menu_or_keep', 'aligned'), ('aligned', 'uniform'))


def check_exchange(data):
    counts, arms, losses = (data[k] for k in ('counts', 'arms', 'losses'))
    if (tuple(counts[k] for k in ('views', 'edits', 'candidates', 'switched')) != (606, 780, 7198, 101)
            or (data['locations'], data['draws']) != (7, 3000)
            or counts['original_passing_options'] != counts['full_grid_passing_options']+losses['any']
            or any(not 0 <= losses[k] <= losses['any'] for k in ('all_risk', 'easy_support', 'easy_excess'))
            or set(arms) != set(ARMS)):
        raise ValueError('Unexpected frozen one-exchange experiment')
    rows = {arm: [arms[arm][field] for field in FIELDS] for arm in ARMS}
    if (any(not 0 <= m <= f <= o <= 606 for o, f, m in rows.values())
            or any(o > bound for arm in ARMS[:-1] for o, bound in zip(rows[arm], rows['menu_or_keep'], strict=True))
            or any(rows[k][i] > rows['mask_oracle'][i] for k in ('uniform', 'aligned') for i in range(len(FIELDS)))):
        raise ValueError('Nested certificate counts are inconsistent')
    return rows


def quantile(values, q):
    i, f = divmod(q*(len(values)-1), 1)
    i = int(i)
    return values[i] + f*(values[min(i+1, len(values)-1)]-values[i])


def replay(data):
    original, sites, draws = (data[k] for k in ('summary', 'sites', 'draws'))
    rows = check_exchange(original)
    if (len(sites) != 7 or any(len(g) != 3 for g in sites)
            or sum(r['views'] for g in sites for r in g) != 606 or len(draws) != 3000
            or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Unexpected source or bootstrap roster')
    for key in ARMS:
        for j, field in enumerate(FIELDS):
            if abs(sum(r['outcomes'][key][field] for g in sites for r in g)-rows[key][j]) > 1e-10:
                raise ValueError('Site counts do not reproduce the aggregate table')
    for a, b in CONTRASTS:
        rates = [mean(100*(r['outcomes'][a]['full_grid']-r['outcomes'][b]['full_grid'])/r['views'] for r in g)
                 for g in sites]
        simulations = sorted(mean(rates[i] for i in draw) for draw in draws)
        v = original['full_grid_contrasts'][f'{a}_minus_{b}']
        if any(abs(got-expect) > 1e-10 for got, expect in zip(
                (mean(rates), quantile(simulations, .025), quantile(simulations, .975)),
                (v['point_pp'], *v['ci95_pp']), strict=True)):
            raise ValueError('Paired source interval did not replay')
    return rows


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement file')
    data = json.loads((root/'exchange.json').read_text())
    result = replay(data)
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        prior = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=prior, opendd_full_grid_exchange=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
