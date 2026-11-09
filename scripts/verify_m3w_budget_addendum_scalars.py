"""Offline scalar replay for the supplementary budget tables."""

import json
from pathlib import Path


def percentile(sorted_values, q):
    index = (len(sorted_values)-1)*q
    lo = int(index)
    hi = min(lo+1, len(sorted_values)-1)
    return sorted_values[lo]+(sorted_values[hi]-sorted_values[lo])*(index-lo)


def replay(data):
    result = {}
    for k, rows in data['counts'].items():
        if len(rows) != 7 or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError('Expected seven aligned source summaries')
        result[k] = [sum(c) for c in zip(*rows, strict=True)]
    draws = data['resamples']
    if len(draws) != 3000 or any(len(d) != 6 or any(not 0 <= i < 6 for i in d) for d in draws):
        raise ValueError('Require six-source bootstrap draws')
    for k, scores in data['contrasts'].items():
        x = [v for v in scores if v is not None]
        if len(x) != 6:
            raise ValueError('A nonestimable source must not be assigned zero')
        boot = sorted(sum(x[i] for i in row)/len(x) for row in draws)
        result[k] = [sum(x)/len(x), percentile(boot, .025), percentile(boot, .975)]
    return result


def validate(data):
    result = replay(data)
    if set(result) != set(data['expected']):
        raise ValueError('Unexpected supplementary table columns')
    for name, values in result.items():
        reference = data['expected'][name]
        if len(values) != len(reference) or any(abs(a-b) > 1e-12 for a, b in zip(values, reference, strict=True)):
            raise ValueError(f'Cannot reproduce {name}')
    return result


if __name__ == '__main__':
    print(json.dumps(validate(json.loads(Path(__file__).with_name('scalars.json').read_text())), indent=2))
