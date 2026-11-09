"""Offline replay of the paper counts and companion decision diagnostics."""

from collections import Counter
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from math import ceil, factorial, floor, fsum, isfinite, prod
from pathlib import Path
import subprocess
import sys


def exact_site_interval(values):
    """Percentiles of all ordered empirical resamples, grouped by multiplicity."""
    n = len(values)
    if n < 2 or not all(isfinite(x) for x in values):
        raise ValueError('A finite estimate is required for every resampling source')
    mass = Counter()
    for draw in combinations_with_replacement(range(n), n):
        weight = factorial(n) // prod(factorial(k) for k in Counter(draw).values())
        mass[fsum(values[i] for i in draw)/n] += weight
    total = n**n
    if sum(mass.values()) != total:
        raise ValueError('Incorrect bootstrap multiplicity')
    ordered = sorted(mass.items())

    def rank(k):
        for v, w in ordered:
            if k < w:
                return v
            k -= w
        raise ValueError('Rank is outside the bootstrap distribution')

    def percentile(q):
        p = (total-1)*q
        lo, hi = rank(floor(p)), rank(ceil(p))
        return lo+(hi-lo)*(p-floor(p))

    return [percentile(.025), percentile(.975)]


def decision(data):
    if (data['locations'], data['recordings'], data['views'], data['seeds']) != (7, 493, 8874, [17, 29, 43]):
        raise ValueError('Unexpected OpenDD development roster')
    out = {}
    for name, contrast in data['contrasts'].items():
        known, lower = contrast['observed'], contrast['lower']
        if len(known) != 7 or len(lower) != 7 or any(
                not isfinite(a) or not isfinite(b) or b > a + 1e-12 for a, b in zip(known, lower, strict=True)):
            raise ValueError('Missing or invalid paired source estimate')
        out[name] = {key: dict(mean=fsum(values)/7, exact_empirical_ci95=exact_site_interval(values))
                     for key, values in (('observed', known), ('lower', lower))}
    return out


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'manifest.json'}
    if actual != set(manifest['members']):
        raise ValueError('Unexpected or missing package members')
    for name, digest in manifest['members'].items():
        if sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed package member: '+name)
    result = {}
    for group in ('headline', 'external', 'budget'):
        p = root/group
        run = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=p,
                             capture_output=True, text=True, check=True, timeout=90)
        result[group] = json.loads(run.stdout)
    if (result['headline']['wepdtof']['fixed_joint_dual_passes'] !=
            result['external']['arms']['fixed_joint']['dual_passes']
            or result['headline']['wepdtof']['views'] != result['external']['views']
            or result['headline']['wepdtof']['sources'] != result['external']['sources']):
        raise ValueError('Headline counts and external pass reconstruction disagree')
    result['opendd_decision_development'] = decision(json.loads((root/'opendd_decisions.json').read_text()))
    return result


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, sort_keys=True, allow_nan=False))
