"""Replay grouped allocation readouts and the original supplement offline."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


MEMBERS = {'README.md', 'allocation.json', 'focused_evidence.zip', 'verify.py'}
ARMS = ('frozen', 'gain_only', 'joint_no_penalty', 'joint_value')


def interval(values, draws):
    estimates = sorted(sum(values[j] for j in d) / len(d) for d in draws)

    def quantile(q):
        at = q * (len(estimates) - 1)
        i = int(at)
        return estimates[i] + (at - i) * (estimates[i + 1] - estimates[i])

    return dict(point=sum(values) / len(values), interval95=[quantile(.025), quantile(.975)],
                positive_locations=sum(v > 0. for v in values))


def replay(data):
    groups, draws = data['locations'], data['bootstrap_location_indices']
    if (len(groups) != 7 or any(len(g) != 3 for g in groups) or len(draws) != 3000
            or any(len(d) != 7 or any(type(j) is not int or not 0 <= j < 7 for j in d) for d in draws)):
        raise ValueError('Expected seven locations, three seeds and 3,000 paired location draws')
    flat = [v for g in groups for v in g]
    for row in flat:
        views = row['views']
        counts = [row[k] for k in ('online_pass', 'count_motion', 'count_only')]
        if not 0 <= counts[0] <= counts[1] <= counts[2] <= views:
            raise ValueError('Inconsistent hindsight nesting')
        for mode in ('online', 'matched'):
            r = row[mode]
            if set(r) != set(ARMS):
                raise ValueError('Expected all frozen and fitted arms')
            for v in r.values():
                if (not 0 <= v['dual_positive'] <= v['risk_pass'] <= views or not 0 <= v['undefined'] <= views
                        or v['edits'] < 0 or not isfinite(v['paired_lower_pp'])):
                    raise ValueError('Inconsistent scored views')
            if mode == 'matched' and any(v['edits'] != row['edits'] for v in r.values()):
                raise ValueError('Matched arm changed the number of edits')
    views, edits = (sum(r[k] for r in flat) for k in ('views', 'edits'))
    if (views, edits) != (606, 780):
        raise ValueError('Unexpected frozen cohort or budget')

    def grouped(fn):
        return interval([sum(fn(r) for r in g) / len(g) for g in groups], draws)

    motion = {k: sum(r[k] for r in flat) for k in ('count_only', 'count_motion')}
    motion['count_motion_minus_online'] = grouped(lambda r: 100. * (r['count_motion'] - r['online_pass']) / r['views'])
    result = dict(views=views, edits=edits, motion=motion)
    for mode in ('online', 'matched'):
        result[mode] = {}
        for arm in ARMS:
            out = {k: sum(r[mode][arm][k] for r in flat) for k in ('edits', 'dual_positive', 'risk_pass', 'undefined')}
            if arm != 'frozen':
                out['paired_pass_rate_pp'] = grouped(lambda r: 100. * (r[mode][arm]['dual_positive']
                    - r[mode]['frozen']['dual_positive']) / r['views'])
                out['paired_conservative_gain_pp'] = grouped(lambda r: r[mode][arm]['paired_lower_pp'])
            result[mode][arm] = out
    return result


def verify(root):
    hashes = json.loads((root / 'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement contents')
    for name, digest in hashes.items():
        p = root / name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed or unsafe bundle member')
    with TemporaryDirectory() as temp, ZipFile(root / 'focused_evidence.zip') as z:
        entries = z.infolist()
        if len({e.filename for e in entries}) != len(entries):
            raise ValueError('Duplicate nested archive names')
        for e in entries:
            p = PurePosixPath(e.filename)
            if (p.is_absolute() or '..' in p.parts or p.as_posix() != e.filename
                    or '\\' in e.filename or e.is_dir() or stat.S_ISLNK(e.external_attr >> 16)):
                raise ValueError('Unsafe path in nested archive')
        z.extractall(temp)
        process = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=temp,
                                 capture_output=True, text=True, timeout=120, check=True)
    inherited = json.loads(process.stdout)
    allocation = replay(json.loads((root / 'allocation.json').read_text()))
    headline = inherited['original']['headline']['opendd']
    if (headline['views'], headline['past_only_passes']) != (allocation['views'], allocation['online']['frozen']['dual_positive']):
        raise ValueError('Original online count does not match the new cohort')
    return dict(inherited=inherited, allocation=allocation)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
