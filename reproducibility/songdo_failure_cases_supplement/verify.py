"""Check the case summaries and replay the inherited anonymous evidence."""

from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'prior_evidence.zip', 'protocol.md', 'cases.json', 'failure_cases.pdf'}
CASES = {'recipients', 'paths', 'missing'}


def close(x, y):
    return isclose(x, y, abs_tol=1e-12, rel_tol=0.)


def replay(row):
    if (set(row['cases']) != CASES or set(row['eligible']) != CASES or row['seed'] != 17
            or row['scored_roster'] != 160 or row['new_cohort_opened'] or row['policy_changed']
            or not row['original_2pct_unchanged'] or row['training']):
        raise ValueError('Unexpected case scope')
    result = {}
    for name, c in row['cases'].items():
        count = row['eligible'][name]
        if not 0 <= count <= row['scored_roster'] or bool(count) != (c is not None):
            raise ValueError('Invalid eligible roster')
        if c is None:
            result[name] = None
            continue
        l, r, v = (c[k] for k in ('left_edits', 'right_edits', 'supported_steps'))
        low, mid, high = (c[k] for k in ('lower', 'observed', 'upper'))
        actor_gain, actor_debit = c['actor_observed'], c['actor_debit']
        if (not 2 <= len(l) == len(r) == len(v) == len(actor_gain) == len(actor_debit) <= 4
                or sum(l) != 2 or sum(r) != 2 or not all(a or b for a, b in zip(l, r))
                or any(type(x) is not bool for x in l+r+[x for mask in v for x in mask])
                or any(len(mask) != 12 for mask in v) or not len(low) == len(mid) == len(high) == 13
                or not all(isfinite(x) for x in low+mid+high+actor_gain+actor_debit)
                or not low[0] == mid[0] == high[0] == 0.
                or not close(fsum(actor_gain), mid[-1]) or not close(fsum(actor_debit), mid[-1]-low[-1])
                or any(d < 0. for d in actor_debit)):
            raise ValueError('Invalid case trace')
        radius = [b-a for a, b in zip(low, mid)]
        if any(not close(a+b, h) for a, b, h in zip(radius, mid, high)):
            raise ValueError('Asymmetric completion bounds')
        for i, (prev, cur) in enumerate(zip(radius, radius[1:])):
            if cur < prev-1e-12 or (all(mask[i] for mask in v) and not close(cur, prev)):
                raise ValueError('Inconsistent missing-label debit')
        if name != 'missing' and not all(all(m) for m in v):
            raise ValueError('Expected completely observed edited actors')
        if name == 'recipients':
            path, allocation = c['path_component_pp'], c['recipient_component_pp']
            if l == r or mid[-1] >= 0. or abs(allocation) < 2*abs(path) or not close(path+allocation, mid[-1]):
                raise ValueError('Invalid recipient example')
        elif name == 'paths' and l != r:
            raise ValueError('Fixed-recipient example changed the actors')
        elif name == 'missing' and not low[-1] < 0 < mid[-1]:
            raise ValueError('Expected an unresolved sign')
        result[name] = dict(eligible=count, observed_pp=mid[-1], lower_pp=low[-1], upper_pp=high[-1])
    if row['numeric_label_joins'] != sum(x is not None for x in result.values()):
        raise ValueError('Unexpected number of numeric joins')
    return result


def verify(root):
    root = Path(root)
    h = json.loads((root/'manifest.json').read_text())
    if set(h) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement contents')
    for k, v in h.items():
        p = root/k
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != v:
            raise ValueError('Changed supplement input')
    row = json.loads((root/'cases.json').read_text())
    if (row['identity']['figure_sha256'] != h['failure_cases.pdf']
            or row['identity']['protocol_sha256'] != h['protocol.md']):
        raise ValueError('Figure, protocol, and summaries are not bound')
    cases = replay(row)
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        parent = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=parent, songdo_failure_cases=cases)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
