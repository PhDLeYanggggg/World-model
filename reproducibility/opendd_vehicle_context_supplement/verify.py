"""Replay anonymous location statistics and the inherited supplement."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json',
       'verify_angular.py', 'vehicle.json'}
MEMBERS = OLD | {'verify.py', 'verify_vehicle.py', 'vehicle_context.json'}


def replay(data):
    sites, draws = data['locations'], data['draws']
    if len(sites) != 7 or len(draws) != 3000 or any(len(d) != 7 or any(
            not isinstance(j, int) or not 0 <= j < 7 for j in d) for d in draws):
        raise ValueError('Invalid location bootstrap')
    k0 = sites[0]['effects'].keys()
    c0 = sites[0]['counts'].keys()
    for row in sites:
        if (row['effects'].keys() != k0 or row['counts'].keys() != c0
                or any(not isfinite(v) for v in row['effects'].values())
                or any(not isinstance(v, int) or v < 0 for v in row['counts'].values())):
            raise ValueError('Invalid anonymous location values')
    counts = {k: sum(r['counts'][k] for r in sites) for k in c0}
    if (counts['views'], counts['original.dual']) != (8874, 2975) or (
            counts['neighbor_model.rotated_views'] != counts['own_at_neighbor.rotated_views']):
        raise ValueError('Frozen recipient roster changed')
    effects = {}
    for key in k0:
        v = [r['effects'][key] for r in sites]
        values = sorted(sum(v[j] for j in d)/7 for d in draws)

        def q(f):
            i = f*(len(values)-1)
            j = int(i)
            return values[j]+(i-j)*(values[j+1]-values[j])

        effects[key] = dict(point=sum(v)/7, ci95=[q(.025), q(.975)])
    return dict(counts=counts, effects=effects)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected vehicle context archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed vehicle context archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_vehicle.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    return dict(inherited=inherited, vehicle_context=replay(json.loads((root/'vehicle_context.json').read_text())))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
