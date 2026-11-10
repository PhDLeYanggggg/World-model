"""Direct completion envelopes and sensitivity to deleting one location."""

from math import fsum, isfinite


def estimate(values, draws):
    n = len(values)
    avg = fsum(values)/n
    samples = sorted(fsum(values[i] for i in row)/n for row in draws)

    def q(p):
        x = p*(len(samples)-1)
        j = int(x)
        return samples[j]+(x-j)*(samples[j+1]-samples[j])

    return dict(point=avg, ci95=[q(.025), q(.975)])


def completion(observed, lower, draws):
    n = len(observed)
    if (n != len(lower) or n < 2 or len(draws) < 2 or not all(isfinite(x) for x in (*observed, *lower))
            or any(l > o+1e-12 for o, l in zip(observed, lower))
            or any(len(row) != n or any(type(i) is not int or not 0 <= i < n for i in row) for row in draws)):
        raise ValueError('Expected aligned direct paired endpoints and source draws')
    upper = [o+(o-l) for o, l in zip(observed, lower)]
    points = dict(observed=observed, lower=lower, upper=upper)
    summary = {k: estimate(x, draws) for k, x in points.items()}
    loo = [[fsum(x[j] for j in range(n) if j != i)/(n-1) for x in (lower, upper)] for i in range(n)]

    def classify(l, u):
        return 'positive' if l > 0. else 'negative' if u < 0. else 'unresolved'

    return dict(summary=summary, leave_one_out_envelopes=loo,
                roster_sign=classify(summary['lower']['point'], summary['upper']['point']),
                leave_one_out_signs=[classify(*r) for r in loo],
                location_signs=[classify(l, u) for l, u in zip(lower, upper)])
