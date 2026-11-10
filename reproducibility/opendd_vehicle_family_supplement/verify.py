"""Replay the family alignment evidence on the fixed nominee roster."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json', 'verify_angular.py',
       'vehicle.json', 'verify_vehicle.py', 'vehicle_context.json', 'verify_context_v11.py',
       'vehicle_action.json', 'verify_action_v12.py', 'vehicle_screen.json', 'verify_screen_v13.py',
       'vehicle_nominee.json'}
MEMBERS = OLD | {'verify.py', 'verify_nominee_v14.py', 'vehicle_family.json'}


def replay(data, previous):
    for site in data['locations']:
        c = site['counts']
        for q in ('ego_count', 'fixed_15pct'):
            if any(c[f'{a}.{q}.count'] != c[f'margin.{q}.count'] for a in ('aligned_pointwise', 'aligned_budget')):
                raise ValueError('Family comparison lost the matched budget')
    return previous(data)


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected family alignment archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_nominee_v14.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        module = runpy.run_path(str(sub/'verify.py'))
        old = module['verify'](sub)
        family = replay(json.loads((root/'vehicle_family.json').read_text()), module['replay'])
    return dict(inherited=old, vehicle_family=family)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
