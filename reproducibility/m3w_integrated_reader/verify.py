"""Verify the consolidated reader tables against the inherited evidence."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'reader_values.json', 'prior_evidence.zip'}


def evidence(tree, key):
    while isinstance(tree, dict):
        if key in tree:
            return tree[key]
        tree = tree.get('inherited')
    raise ValueError(f'Inherited evidence missing: {key}')


def near(a, b):
    return isclose(a, b, rel_tol=0., abs_tol=1e-10)


def compare(data, parent):
    uniform = evidence(parent, 'opendd_fixed_spend_uniform')
    fixed = evidence(parent, 'opendd_full_grid_count_motion')
    listwise = evidence(parent, 'opendd_listwise_full_grid')
    relative = evidence(parent, 'opendd_backbone_relative')
    masking = evidence(parent, 'opendd_pair_label_stress')
    if not (data['uniform'] == uniform and data['count'] == fixed and data['relative'] == relative
            and data['masking'] == masking and set(data['listwise']) == set(listwise)
            and all(near(data['listwise'][k], v) for k, v in listwise.items())):
        raise ValueError('Reader tables do not match the inherited aggregates')
    return dict(views=fixed['views'], edits=fixed['edits'], full_grid=fixed['full_grid'],
                uniform_expected=uniform['expected_passes_low'], complete_queries=masking['support']['queries'],
                relative_pair_steps=relative['support']['pair_steps'])


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected reviewer archive members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed reviewer archive member')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for name in names:
            (folder/name).write_bytes(z.read(name))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    values = json.loads((root/'reader_values.json').read_text())
    return dict(inherited=parent, integrated_reader=compare(values, parent))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
