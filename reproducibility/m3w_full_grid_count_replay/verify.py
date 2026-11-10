"""Replay fixed-spend full-grid counts and source-paired intervals."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'count_motion.json', 'prior_evidence.zip'}
FIELDS = ('views', 'edits', 'archived_positive', 'archived_unresolved', 'new_original_witness',
          'original_verified_after_search', 'blind_full_grid', 'one_exchange_full_grid', 'full_grid', 'unresolved')
CONTROLS = ('blind_full_grid', 'original_verified_after_search', 'one_exchange_full_grid')


def percentile(z, p):
    i, f = divmod(p*(len(z)-1), 1)
    i = int(i)
    return z[i] + f*(z[i+1]-z[i])


def replay(data):
    counts, groups, draws = (data[k] for k in ('counts', 'locations', 'draws'))
    if (set(counts) != set(FIELDS) or len(groups) != 7 or any(len(g) != 3 for g in groups)
            or len(draws) != 3000 or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)
            or (counts['views'], counts['edits'], counts['unresolved']) != (606, 780, 0)
            or any(counts[k] != sum(r[k] for g in groups for r in g) for k in FIELDS)
            or counts['archived_unresolved'] != counts['new_original_witness']
            or counts['original_verified_after_search'] != counts['archived_positive']+counts['new_original_witness']
            or not counts['blind_full_grid'] <= counts['one_exchange_full_grid'] <= counts['full_grid'] <= counts['original_verified_after_search']):
        raise ValueError('Count-and-motion feasibility arithmetic differs')
    if set(data['contrasts']) != {f'full_grid_minus_{r}' for r in CONTROLS}:
        raise ValueError('Unexpected contrasts')
    for r in CONTROLS:
        rate = [mean(100*(v['full_grid']-v[r])/v['views'] for v in g) for g in groups]
        samp = sorted(mean(rate[i] for i in draw) for draw in draws)
        ref = data['contrasts'][f'full_grid_minus_{r}']
        if any(abs(a-b) > 1e-10 for a, b in zip((mean(rate), percentile(samp, .025), percentile(samp, .975)),
                                               (ref['point_pp'], *ref['ci95_pp']), strict=True)):
            raise ValueError('Paired source interval differs')
    return counts


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement file')
    counts = replay(json.loads((root/'count_motion.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        if (root/'supplement.pdf').read_bytes() != (folder/'supplement.pdf').read_bytes():
            raise ValueError('The reader PDF does not match the inherited evidence')
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    rows = parent['inherited']['opendd_full_grid_exchange']
    if rows['menu_or_keep'][1] != counts['one_exchange_full_grid'] or rows['incumbent'][1] != counts['blind_full_grid']:
        raise ValueError('Matched controls did not reproduce the parent')
    return dict(inherited=parent, opendd_full_grid_count_motion=counts)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
