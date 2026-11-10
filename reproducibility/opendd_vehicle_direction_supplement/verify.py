"""Replay the fixed-recipient vehicle direction results and prior evidence."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json'}
MEMBERS = OLD | {'verify.py', 'verify_angular.py', 'vehicle.json'}


def replay(data):
    draws, panels = data['draws'], data['panels']
    if (set(panels) != {'forced', 'one_rotation'} or len(draws) != 3000 or any(
            len(d) != 7 or any(not 0 <= i < 7 for i in d) for d in draws)):
        raise ValueError('Invalid vehicle bootstrap roster')
    out = {}
    for k, groups in panels.items():
        if len(groups) != 7 or any(len(g) != 3 or any(len(seed) != 6 for seed in g) for g in groups):
            raise ValueError('Expected seven locations by three seeds by six directions')
        flat = [r for g in groups for s in g for r in s]
        c, v = flat[0]['counts'], flat[0]['effect']
        if any(r['counts'].keys() != c.keys() or r['effect'].keys() != v.keys()
               or any(not isfinite(x) or x < 0 for x in r['counts'].values())
               or any(not isfinite(x) for x in r['effect'].values()) for r in flat):
            raise ValueError('Unexpected group fields')
        counts = {a: sum(r['counts'][a] for r in flat) for a in c}
        if (counts['views'], counts['original_dual']) != (8874, 2975):
            raise ValueError('Frozen vehicle cohort changed')
        effect = {}
        for a in v:
            means = [sum(sum(r['effect'][a] for r in s)/len(s) for s in g)/len(g) for g in groups]
            bootstrap = sorted(sum(means[i] for i in d)/len(d) for d in draws)

            def q(f):
                pos = f*(len(bootstrap)-1)
                j = int(pos)
                return bootstrap[j]+(pos-j)*(bootstrap[j+1]-bootstrap[j])

            effect[a] = dict(point=sum(means)/len(means), ci95=[q(.025), q(.975)])
        out[k] = dict(counts=counts, effects=effect)
    if out['one_rotation']['counts']['rotations'] != out['one_rotation']['counts']['matched_rotations']:
        raise ValueError('One-rotation comparison lost the shared mask')
    return out


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected vehicle direction archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed vehicle direction archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_angular.py').read_bytes())
        manifest = {p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}
        (sub/'manifest.json').write_text(json.dumps(manifest))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=inherited, vehicle_direction=replay(json.loads((root/'vehicle.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
