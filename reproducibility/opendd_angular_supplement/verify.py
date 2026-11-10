"""Reproduce direction and threshold results from anonymous group summaries."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json'}
MEMBERS = OLD | {'verify.py', 'verify_context.py', 'angular.json'}
PANELS = {'direction_menu', 'same_recipients', 'angle_oracle', 'angle_choice', 'calibration', 'matched_mask'}


def replay(panels, draws):
    if set(panels) != PANELS or len(draws) != 3000 or any(
            len(d) != 7 or any(not 0 <= i < 7 for i in d) for d in draws):
        raise ValueError('Unexpected panel or location-bootstrap contract')
    result = {}
    for label, groups in panels.items():
        if len(groups) != 7 or any(len(g) != 3 for g in groups):
            raise ValueError('Expected seven locations and three seeds')
        rows = [r for group in groups for r in group]
        c, p = rows[0]['counts'], rows[0]['effects']
        if any(r['counts'].keys() != c.keys() or r['effects'].keys() != p.keys()
               or any(not isfinite(v) or v < 0 for v in r['counts'].values())
               or any(not isfinite(v) for v in r['effects'].values()) for r in rows):
            raise ValueError('Unexpected group summary fields')
        counts = {k: sum(r['counts'][k] for r in rows) for k in c}
        if (counts['views'], counts['edits']) != (606, 780):
            raise ValueError('Original roster changed')
        effect = {}
        for k in p:
            mean = [sum(r['effects'][k] for r in g)/len(g) for g in groups]
            draws_sorted = sorted(sum(mean[i] for i in d)/len(d) for d in draws)

            def percentile(q):
                p = q*(len(draws_sorted)-1)
                j = int(p)
                return draws_sorted[j]+(p-j)*(draws_sorted[j+1]-draws_sorted[j])

            effect[k] = dict(point_pp=sum(mean)/len(mean), ci95_pp=[percentile(.025), percentile(.975)])
        result[label] = dict(counts=counts, effects=effect)
    c = result['calibration']['counts']
    m = result['matched_mask']['counts']
    if c['calibrated_rotations'] != m['rotations'] or c['constant_rotations'] != m['rotations']:
        raise ValueError('Matched rotation counts changed')
    return result


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected angular supplement contents')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed angular supplement member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_context.py').read_bytes())
        manifest = {p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}
        (sub/'manifest.json').write_text(json.dumps(manifest))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    draws = json.loads((root/'frontier.json').read_text())['draws']
    angular = replay(json.loads((root/'angular.json').read_text()), draws)
    return dict(inherited=inherited, angular=angular)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
