"""Check the reader supplement and replay anonymous statistics offline."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'risk.json', 'prior_evidence.zip'}


def check_risk(data):
    if (data['flights'], data['components'], data['seeds']) != (160, 14, [17, 29, 43]):
        raise ValueError('Unexpected confirmation roster')
    if set(data['results']) != {'independent', 'additive', 'same_family', 'joint'}:
        raise ValueError('Unexpected policy roster')
    out = {}
    for a, v in data['results'].items():
        m, r = v['matched'], v['actual_releases']
        total = r['total']
        if (m['total'] != 480 or not 0 <= total <= 480 or total+v['abstained'] != 480
                or m['original_three_checks'] != m['original_three_and_full_grid_all']+m['original_three_with_adverse_completion']
                or not 0 <= m['original_three_with_adverse_completion'] <= m['original_three_with_partial_or_unknown']
                or not 0 <= m['original_three_and_full_grid_all'] <= m['full_grid_all_pass'] <= 480):
            raise ValueError('Inconsistent matched and released risk counts')
        if total and (r['partial_or_unknown'] or r['full_grid_all_pass'] != r['original_all_pass']
                      or r['original_three_checks'] != r['original_three_and_full_grid_all']):
            raise ValueError('Expected complete labels for the actually edited tracks')
        out[a] = [m['original_three_checks'], m['original_three_and_full_grid_all'], total]
    return out


def verify(root):
    root = Path(root)
    h = json.loads((root/'manifest.json').read_text())
    if set(h) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for k, v in h.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed supplement file')
    result = check_risk(json.loads((root/'risk.json').read_text()))
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, full_grid_all_risk=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
