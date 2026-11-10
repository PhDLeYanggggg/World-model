"""Replay the frozen ranking and check the inherited anonymous supplement."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'prior_evidence.zip', 'protocol.md', 'rank_evidence.json',
           'readout.json', 'frozen_score_ranking.py', 'replay_songdo.py'}


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement contents')
    for name, digest in manifest.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement member')
    original = SimpleNamespace(**runpy.run_path(str(root/'replay_songdo.py')))
    rank = runpy.run_path(str(root/'frozen_score_ranking.py'))
    evidence = json.loads((root/'rank_evidence.json').read_text())
    result = json.loads((root/'readout.json').read_text())
    if (result['songdo_scalar_sha256'] != manifest['rank_evidence.json']
            or result['protocol_sha256'] != manifest['protocol.md'] or result['new_labels_read']
            or result['threshold_changed'] or result['new_confirmation'] or result['deployment_changed']
            or result['original_risk_limit'] != original.RISK_LIMIT):
        raise ValueError('Changed analysis scope')
    frozen = original.replay(evidence)
    actual = rank['diagnostic'](evidence, frozen, original)
    if actual != result['arms']:
        raise ValueError('Ranking results do not replay')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=inherited, songdo_score_ranking=actual)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
