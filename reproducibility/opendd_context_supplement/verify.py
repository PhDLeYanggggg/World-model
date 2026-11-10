"""Replay matched context selection after verifying the earlier evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


OLD = {'README.md', 'frontier.json', 'decision_evidence.zip', 'ranking.json', 'verify_frontier.py'}
MEMBERS = OLD | {'verify_exchange.py', 'verify.py', 'context.json'}
MODES = ('motion', 'map', 'peer', 'aligned', 'misaligned')
FIELDS = ('dual_positive', 'risk_pass', 'undefined')


def replay(groups, draws):
    if len(groups) != 7 or any(len(g) != 3 for g in groups):
        raise ValueError('Expected seven locations and three seeds')
    rows = [r for g in groups for r in g]
    for r in rows:
        if set(r['own_mask']) != set(MODES) | {'frozen'} or set(r['matched']) != set(MODES) | {'random'}:
            raise ValueError('Context control roster changed')
        if not 0 <= r['exchanged'] <= r['views'] or any(r['matched']['aligned'][f] != r['own_mask']['aligned'][f] for f in FIELDS):
            raise ValueError('Common swap mask changed aligned outcomes')
        for k in ('own_mask', 'matched'):
            for s in r[k].values():
                if not 0 <= s['dual_positive'] <= s['risk_pass'] <= r['views'] or not 0 <= s['undefined'] <= r['views']:
                    raise ValueError('Invalid context risk count')
    out = {k: sum(r[k] for r in rows) for k in ('views', 'edits', 'exchanged')}
    if (out['views'], out['edits']) != (606, 780):
        raise ValueError('Matched context budget changed')
    out |= {k: {a: {f: sum(r[k][a][f] for r in rows) for f in FIELDS} for a in rows[0][k]} for k in ('own_mask', 'matched')}

    def interval(fn):
        x = [sum(fn(r) for r in g)/len(g) for g in groups]
        estimates = sorted(sum(x[i] for i in d)/len(d) for d in draws)

        def quantile(q):
            z = q*(len(estimates)-1)
            i = int(z)
            return estimates[i]+(z-i)*(estimates[i+1]-estimates[i])

        return dict(point_pp=sum(x)/len(x), ci95_pp=[quantile(.025), quantile(.975)])

    out['aligned_minus'] = {}
    for a in ('motion', 'misaligned', 'random'):
        out['aligned_minus'][a] = dict(gain_pp=interval(lambda r: r['effect'][a]),
            dual_pp=interval(lambda r: 100.*(r['matched']['aligned']['dual_positive']-r['matched'][a]['dual_positive'])/r['views']))
    return out


def verify(root):
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected context supplement contents')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed context supplement member')
    with TemporaryDirectory() as directory:
        t = Path(directory)
        for k in OLD:
            (t/k).write_bytes((root/k).read_bytes())
        (t/'verify.py').write_bytes((root/'verify_exchange.py').read_bytes())
        (t/'manifest.json').write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in t.iterdir()}))
        inherited = runpy.run_path(str(t/'verify.py'))['verify'](t)
    x = json.loads((root/'frontier.json').read_text())
    context = replay(json.loads((root/'context.json').read_text())['locations'], x['draws'])
    if inherited['exact_exchange']['anchor_dual'] != context['own_mask']['frozen']['dual_positive']:
        raise ValueError('Earlier no-exchange reference did not reproduce')
    return dict(inherited=inherited, exchange_context=context)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
