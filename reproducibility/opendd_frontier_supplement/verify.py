"""Recompute the exchange frontier without checkpoints or future coordinates."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


MEMBERS = {'README.md', 'frontier.json', 'decision_evidence.zip', 'verify.py'}
LEVELS = ('zero', 'one', 'two', 'unrestricted')
FLOORS = ('positive', '0.10pp', '0.50pp', '1.00pp')


def replay(data, expected_anchor):
    groups, draws = data['locations'], data['draws']
    if (len(groups) != 7 or any(len(g) != 3 for g in groups) or len(draws) != 3000
            or any(len(d) != 7 or any(type(i) is not int or not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Expected seven locations, three seeds and 3,000 bootstrap draws')
    rows = [r for g in groups for r in g]
    for row in rows:
        if (set(row['bounds']) != set(LEVELS) or any(set(a) != set(FLOORS) for a in row['bounds'].values())
                or row['bounds']['zero']['positive'] != [row['anchor_dual']] * 2):
            raise ValueError('Unexpected frontier levels or anchor')
        for floor in FLOORS:
            lo, hi = zip(*(row['bounds'][k][floor] for k in LEVELS), strict=True)
            if (any(type(n) is not int or not 0 <= n <= row['views'] for n in (*lo, *hi))
                    or any(a > b for a, b in zip(lo, hi, strict=True)) or list(lo) != sorted(lo) or list(hi) != sorted(hi)):
                raise ValueError('Invalid or nonnested frontier counts')
    totals = tuple(sum(r[k] for r in rows) for k in ('views', 'edits', 'anchor_dual'))
    if totals != (606, 780, expected_anchor):
        raise ValueError('Inherited cohort or edited count changed')
    counts = {k: {f: [sum(r['bounds'][k][f][i] for r in rows) for i in range(2)] for f in FLOORS} for k in LEVELS}
    point = [sum(100. * (r['bounds']['one']['positive'][0]-r['anchor_dual'])/r['views'] for r in g)/len(g) for g in groups]
    boot = sorted(sum(point[i] for i in draw)/len(draw) for draw in draws)

    def quantile(q):
        x = (len(boot)-1)*q
        i = int(x)
        return boot[i]+(x-i)*(boot[i+1]-boot[i])

    return dict(views=totals[0], edits=totals[1], anchor_dual=totals[2], bounds=counts,
        verified_one_minus_zero_pp=dict(point_pp=sum(point)/len(point), ci95_pp=[quantile(.025), quantile(.975)],
                                        locations_positive=sum(v > 0. for v in point)))


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected frontier supplement contents')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed frontier supplement member')
    with TemporaryDirectory() as temp, ZipFile(root/'decision_evidence.zip') as z:
        entries = z.infolist()
        if len({e.filename for e in entries}) != len(entries):
            raise ValueError('Duplicate inherited archive names')
        for e in entries:
            p = PurePosixPath(e.filename)
            if (p.is_absolute() or '..' in p.parts or p.as_posix() != e.filename
                    or '\\' in e.filename or e.is_dir() or stat.S_ISLNK(e.external_attr >> 16)):
                raise ValueError('Unsafe inherited archive path')
        z.extractall(temp)
        p = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=temp,
                           capture_output=True, text=True, timeout=120, check=True)
    inherited = json.loads(p.stdout)
    anchor = inherited['decision_controls']['arms']['frozen']['dual_positive']
    frontier = replay(json.loads((root/'frontier.json').read_text()), anchor)
    return dict(inherited=inherited, exchange_frontier=frontier)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
