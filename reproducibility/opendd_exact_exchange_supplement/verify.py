"""Replay the exact exchange readout and the inherited frontier evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip'}
MEMBERS = OLD | {'verify_frontier.py', 'verify.py', 'ranking.json'}
ARMS = ('context_guarded', 'context_forced')


def replay(groups, frontier, draws, interval):
    if len(groups) != 7 or any(len(g) != 3 for g in groups):
        raise ValueError('Expected seven locations and three seeds')
    flat = [r for g in groups for r in g]
    fields = ('views', 'edits', 'candidates', 'anchor_dual')
    result = {k: sum(r[k] for r in flat) for k in fields}
    if tuple(result.values()) != (606, 780, 7198, frontier['anchor_dual']):
        raise ValueError('Exhaustive menu changed the inherited evaluation')
    result['exact_frontier'] = {}
    for k, bounds in frontier['bounds']['one'].items():
        v = sum(r['exact_frontier'][k] for r in flat)
        if not bounds[0] <= v <= bounds[1]:
            raise ValueError('Exhaustive score contradicts the archived frontier')
        result['exact_frontier'][k] = v

    def difference(arm, key):
        values = []
        for g in groups:
            p = []
            for r in g:
                c = r['comparisons'][arm]
                x = c['chosen_' + key] - c['random_' + key]
                p.append(100.*x/r['views'] if key == 'dual' else x)
            values.append(sum(p)/len(p))
        return interval(values, draws)

    result['comparisons'] = {}
    for arm in ARMS:
        for r in flat:
            c = r['comparisons'][arm]
            n, a = r['views'], r['anchor_dual']
            if (not 0 <= c['exchanged'] <= n or c['chosen_dual'] != a+c['rescues']-c['destroys']
                    or abs(c['random_dual']-a-c['random_rescued']+c['random_destroyed']) > 1e-10
                    or not 0 <= c['chosen_dual'] <= n or not 0 <= c['random_dual'] <= n):
                raise ValueError('Chance or chosen transitions do not reconcile')
        keys = ('chosen_dual', 'random_dual', 'random_rescued', 'random_destroyed', 'rescues', 'destroys', 'exchanged')
        result['comparisons'][arm] = {k: sum(r['comparisons'][arm][k] for r in flat) for k in keys}
        result['comparisons'][arm].update(dual_rate_minus_random_pp=difference(arm, 'dual'),
                                           paired_gain_minus_random_pp=difference(arm, 'gain_pp'))
    return result


def verify(root):
    expected = json.loads((root/'manifest.json').read_text())
    if set(expected) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected exchange supplement contents')
    for name, digest in expected.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed exchange supplement member')
    with TemporaryDirectory() as directory:
        t = Path(directory)
        for k in OLD:
            (t/k).write_bytes((root/k).read_bytes())
        (t/'verify.py').write_bytes((root/'verify_frontier.py').read_bytes())
        hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in t.iterdir()}
        (t/'manifest.json').write_text(json.dumps(hashes))
        prior = runpy.run_path(str(t/'verify.py'))
        inherited = prior['verify'](t)
        draws = json.loads((t/'frontier.json').read_text())['draws']
        allocation = root/'ranking.json'

        def interval(x, draws):
            vals = sorted(sum(x[i] for i in d)/len(d) for d in draws)

            def quantile(q):
                v = (len(vals)-1)*q
                j = int(v)
                return vals[j]+(v-j)*(vals[j+1]-vals[j])

            return dict(point_pp=sum(x)/len(x), ci95_pp=[quantile(.025), quantile(.975)],
                        locations_positive=sum(v > 0. for v in x))

        exact = replay(json.loads(allocation.read_text())['locations'], inherited['exchange_frontier'], draws, interval)
    return dict(inherited=inherited, exact_exchange=exact)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
