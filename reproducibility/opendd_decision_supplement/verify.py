"""Recompute the decision controls after verifying the inherited evidence."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import runpy
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


MEMBERS = {'README.md', 'decision_controls.json', 'allocation_evidence.zip', 'verify.py'}
ARMS = ('frozen', 'pointwise', 'original_pairwise', 'continued_pairwise', 'conditional_utility')
FIELDS = ('edits', 'dual_positive', 'risk_pass', 'undefined')
PRIMARY = 'conditional_utility_minus_continued_pairwise'


def replay(groups, earlier, draws, interval):
    if len(groups) != len(earlier) or len(groups) != 7 or any(len(g) != 3 for g in groups):
        raise ValueError('Expected seven matched locations and three seeds')
    for rows, parents in zip(groups, earlier, strict=True):
        for row, ref in zip(rows, parents, strict=True):
            if ((row['views'], row['scores']['frozen']) != (ref['views'], {k: ref['matched']['frozen'][k] for k in FIELDS})
                    or set(row['scores']) != set(ARMS) or not isfinite(row['paired_conservative_gain_pp'])):
                raise ValueError('Incompatible inherited roster or frozen score')
            for s in row['scores'].values():
                if (s['edits'] != ref['edits'] or not 0 <= s['dual_positive'] <= s['risk_pass'] <= row['views']
                        or not 0 <= s['undefined'] <= row['views']):
                    raise ValueError('Matched arm changed budget or risk counts')

    def grouped(fn):
        return interval([sum(fn(r) for r in g) / len(g) for g in groups], draws)

    counts = {arm: {f: sum(r['scores'][arm][f] for g in groups for r in g) for f in FIELDS} for arm in ARMS}
    return dict(arms=counts, primary={
        'pass_rate_pp': grouped(lambda r: 100. * (r['scores']['conditional_utility']['dual_positive']
            - r['scores']['continued_pairwise']['dual_positive']) / r['views']),
        'conservative_gain_pp': grouped(lambda r: r['paired_conservative_gain_pp'])})


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected decision supplement contents')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed decision supplement member')
    with TemporaryDirectory() as temp, ZipFile(root/'allocation_evidence.zip') as z:
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
        allocation = json.loads((Path(temp)/'allocation.json').read_text())
        interval = runpy.run_path(str(Path(temp)/'verify.py'))['interval']
        outcome = replay(json.loads((root/'decision_controls.json').read_text())['locations'], allocation['locations'],
                         allocation['bootstrap_location_indices'], interval)
    return dict(inherited=inherited, decision_controls=outcome)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
