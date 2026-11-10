"""Replay external completion envelopes separately from sampling uncertainty."""

from hashlib import sha256
from io import BytesIO
import json
from math import fsum, isfinite
from pathlib import Path
import runpy
from statistics import quantiles
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'external_completion.json', 'prior_evidence.zip'}


def mean(v):
    return fsum(v)/len(v)


def replay(d):
    draws, source = d['draws'], d['songdo_components']
    if set(source) != {'independent', 'additive', 'same_family'} or len(draws) != 3000 or any(
            len(s) != 14 or any(type(i) is not int or not 0 <= i < 14 for i in s) for s in draws):
        raise ValueError('Invalid paired source bootstrap')
    result = {}
    for a, r in source.items():
        lo, seen = r['lower'], r['observed']
        if len(lo) != 14 or len(seen) != 14 or any(
                not isfinite(x) or not isfinite(y) or x > y+1e-12 for x, y in zip(lo, seen, strict=True)):
            raise ValueError('Invalid componentwise debit')
        upper = [2*y-x for x, y in zip(lo, seen, strict=True)]
        row = {}
        for label, values in (('lower', lo), ('observed', seen), ('upper', upper)):
            bounds = quantiles([mean([values[i] for i in s]) for s in draws], n=40, method='inclusive')
            row.update({label: mean(values), f'{label}_ci95': [bounds[0], bounds[-1]]})
        leave = [(fsum(upper)-v)/13 for v in upper]
        row.update(leave_one_component_out_upper_range=[min(leave), max(leave)],
                   components_with_positive_lower=sum(x > 0 for x in lo),
                   components_with_negative_upper=sum(x < 0 for x in upper))
        result[a] = row
    return result


def locate_wepdtof(row):
    if not isinstance(row, dict):
        return None
    if {'contrasts', 'matched_estimable_sources', 'original_fixed_joint_dual_passes', 'views'} <= row.keys():
        return row
    for value in row.values():
        result = locate_wepdtof(value)
        if result is not None:
            return result
    return None


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected archive members')
    for name, expected in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed evidence member')
    with TemporaryDirectory() as tmp:
        sub = Path(tmp)
        with ZipFile(BytesIO((root/'prior_evidence.zip').read_bytes())) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
                raise ValueError('Invalid inherited filename')
            for n in names:
                (sub/n).write_bytes(z.read(n))
        inherited = runpy.run_path(str(sub/'verify.py'))['verify'](sub)
    wep = locate_wepdtof(inherited)
    if wep is None or (wep['views'], wep['matched_estimable_sources'], wep['original_fixed_joint_dual_passes']) != (45, 10, 0):
        raise ValueError('Inherited WEPDTOF contrast changed')
    raw = json.loads((root/'external_completion.json').read_text())
    env = wep['contrasts']['fixed_independent']['identification_envelope_pp']
    if any(abs(a-b) > 1e-12 for a, b in zip(env, raw['wepdtof_bounds'], strict=True)):
        raise ValueError('WEPDTOF missing-label envelope did not replay')
    return dict(inherited=inherited, external_completion=dict(songdo=replay(raw), wepdtof_bounds=env))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
