"""Replay grouped action-value statistics and the inherited evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py',
       'verify_exchange.py', 'context.json', 'verify_context.py', 'angular.json', 'verify_angular.py',
       'vehicle.json', 'verify_vehicle.py', 'vehicle_context.json'}
MEMBERS = OLD | {'verify.py', 'verify_context_v11.py', 'vehicle_action.json'}


def replay(data, grouped_replay):
    # Reuse the location bootstrap with renamed mask keys; also check both extra controls.
    expected = ('ego_at_context', 'constant_at_context', 'joint_at_context')
    clean = []
    for loc in data['locations']:
        c = loc['counts']
        n = c['context.rotations']
        if any(c[f'{a}.rotations'] != n for a in expected):
            raise ValueError('Action-value mask changed')
        clean.append(dict(effects=loc['effects'], counts={**c, 'neighbor_model.rotated_views': n,
                                                           'own_at_neighbor.rotated_views': n}))
    out = grouped_replay(dict(locations=clean, draws=data['draws']))
    for k in ('neighbor_model.rotated_views', 'own_at_neighbor.rotated_views'):
        del out['counts'][k]
    return out


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected action-value archive')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed action-value archive member')
    with TemporaryDirectory() as directory:
        sub = Path(directory)
        for name in OLD:
            (sub/name).write_bytes((root/name).read_bytes())
        (sub/'verify.py').write_bytes((root/'verify_context_v11.py').read_bytes())
        (sub/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sub.iterdir()}))
        module = runpy.run_path(str(sub/'verify.py'))
        inherited = module['verify'](sub)
        action = replay(json.loads((root/'vehicle_action.json').read_text()), module['replay'])
    return dict(inherited=inherited, vehicle_action=action)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
