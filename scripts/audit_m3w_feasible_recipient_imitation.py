"""Separate lost recipient opportunity from shape choices on frozen paths."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_feasible_recipient_imitation as exp

run, np = exp.run, exp.np


def restrict(p, path, cv):
    if not len(p['rows']):
        return p
    r = p['rows']
    peak = exp.control.radius.peak
    a = peak(path[r]-cv[r])
    keep = (a > 0.) & (np.abs(peak(p['paths']-cv[r])-a) < 1e-9)
    match = (np.abs(p['paths']-path[r]).max((1, 2)) < 1e-10) & keep
    selected = exp.decode(p, np.where(match, 1., -1.))
    if not match[selected].all():
        raise ValueError('Frozen forecast cannot be reconstructed at its original slots')
    return dict(p, rows=r[keep], slots=p['slots'][keep], paths=p['paths'][keep],
                anchors=np.searchsorted(np.flatnonzero(keep), selected))


def main():
    reg, source, *_ = exp.register()
    fp, rp = (exp.OUT/k for k in ('freeze.json', 'readout.json'))
    f, r = map(run.read, (fp, rp))
    if (f['registration'] != reg or r['freeze'] != run.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or r['confirmation_access'] or r['independent_confirmation']):
        raise ValueError('Require frozen development predictions')
    data = exp.m.load(source)
    arms = ('anchor', 'prefix_match', *exp.ARMS)
    cells = []
    for c in f['cells']:
        d = data[c['held']]['d']
        bank = exp.control.parent.prior.analytic(d)
        cv, ols = bank[:2]
        with np.load(run.checked(c['paths']), allow_pickle=False) as z:
            paths = {k:z[k] for k in arms}
        with np.load(run.checked(c['spline']), allow_pickle=False) as z:
            p = exp.design(d, bank, paths['anchor'], z['comoving_spline'])
        stats = {}
        for k, a in paths.items():
            q = restrict(p, a, cv) if k != 'prefix_match' else restrict(p, paths['anchor'], cv)
            _, counts = exp.labels(d, bank, q, c['cut'], ('risk_teacher',))
            y, valid, rec = d['y'], d['valid'], d['rec']
            obs = exp.prior.policy(a, cv, cv, y, valid, rec, easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            misses = Counter()
            for w in obs['recordings'].values():
                for field in ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation'):
                    t = w[field]
                    misses[field+('_undefined' if t is None else '_above_limit' if t > .02 else '_within_limit')] += 1
            stats[k] = dict(capacity=counts['risk_teacher'], risk_components=dict(misses))
        cells.append(dict(site=c['held'], seed=c['seed'], arms=stats))
    totals = {k: {field:dict(sum((Counter(c['arms'][k][field]) for c in cells), Counter()))
               for field in ('capacity', 'risk_components')} for k in arms}
    run.once(exp.OUT/'attribution.json', run.stable(dict(parent=run.artifact(rp), code=run.artifact(Path(__file__)),
        cells=cells, summary=totals, label_aware_only=True, confirmation_access=False, original_2pct=True)))
    print(totals, flush=True)


if __name__ == '__main__':
    main()
