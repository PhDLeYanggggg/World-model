"""Replay the recording screen and the inherited anonymous evidence."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json', 'verify_angular.py',
       'vehicle.json', 'verify_vehicle.py', 'vehicle_context.json', 'verify_context_v11.py', 'vehicle_action.json'}
MEMBERS = OLD | {'verify.py', 'verify_action_v12.py', 'vehicle_screen.json'}


def replay(data):
    sites, draws = data['locations'], data['draws']
    if len(sites) != 7 or len(draws) != 3000 or any(len(d) != 7 or any(
            not isinstance(i, int) or not 0 <= i < 7 for i in d) for d in draws):
        raise ValueError('Invalid location bootstrap')
    counts, effects = sites[0]['counts'].keys(), sites[0]['effects'].keys()
    for row in sites:
        c, e = row['counts'], row['effects']
        if (c.keys() != counts or e.keys() != effects or any(not isinstance(v, int) or v < 0 for v in c.values())
                or any(not isfinite(v) for v in e.values()) or not 0 <= c['nominee_strict'] <= c['exists'] <= c['views']):
            raise ValueError('Invalid anonymous site aggregate')
        for quota in ('ego_count', 'fixed_15pct'):
            if len({c[f'{a}.{quota}.count'] for a in ('margin', 'opportunity', 'product')}) != 1:
                raise ValueError('Changed equal-count budget')
    n = {k: sum(r['counts'][k] for r in sites) for k in counts}
    if (n['views'], n['exists'], n['original_dual']) != (8874, 1393, 2975):
        raise ValueError('Changed fixed roster')
    p = {}
    for key in effects:
        m = [r['effects'][key] for r in sites]
        b = sorted(sum(m[i] for i in d)/7 for d in draws)

        def q(f):
            at = (len(b)-1)*f
            i = int(at)
            return b[i]+(at-i)*(b[i+1]-b[i])

        p[key] = dict(point=sum(m)/7, ci95=[q(.025), q(.975)])
    return dict(counts=n, effects=p)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected screen archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_action_v12.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=inherited, vehicle_screen=replay(json.loads((root/'vehicle_screen.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
