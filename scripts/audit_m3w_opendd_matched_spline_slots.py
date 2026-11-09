"""Compare spline and EqMotion shapes at unchanged OpenDD recipients and radii."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_matched_spline as m

control, run, np = m.control, m.run, m.np


def main():
    f, r = (m.OUT / n for n in ('freeze.json', 'readout.json'))
    freeze, result = map(run.read, (f, r))
    if (result['freeze'] != run.sha(f) or not freeze['before_held_scoring']
            or result['confirmation_access'] or result['independent_confirmation'] or not result['original_2pct']):
        raise ValueError('Require frozen held development predictions')
    _, _, ph, h, anchor, source = control.register()
    reference = control.OUT/'analysis.json'
    eq = run.read(reference)
    reg = m.OUT/'slot_registration.json'
    run.once(reg, run.stable(dict(parents=[run.artifact(p) for p in (f, r, reference)],
        code=run.artifact(Path(__file__)), primary='comoving spline versus original at identical recipients and peaks',
        original_2pct=True, exploratory_after_direct_readout=True, confirmation_access=False,
        no_new_model_selection=True, label_aware_capacity_diagnostic_only=True)))
    sites = {s: control.load(s, source) for s in control.parent.source.SITES}
    rows = []
    for held, (d, rec, starts, *_rest) in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s][0]['cut_samples'] for s in donors]), [.25, .75])
        bank = control.parent.prior.analytic(d)
        packet = control.prior.prepare({s: z[0] for s, z in sites.items()}, held, donors, bank)
        for cell in (x for x in freeze['cells'] if x['held'] == held):
            seed = cell['seed']
            ref = next(v for v in anchor['by_site'][held] if v['seed'] == seed)
            past = next(v for v in eq['by_site'][held] if v['seed'] == seed)['matched']
            with (np.load(run.checked(cell['paths']), allow_pickle=False) as z,
                  np.load(run.checked(cell['eqmotion_paths']), allow_pickle=False) as old):
                p = {**{k: z[k] for k in m.model.MODES}, 'comoving_ego': old['comoving_ego']}
            menu, count = control.ledger.frozen_count(d, bank, rec, starts,
                ph['heads'][f'{held}/{seed}/additive_rank'], h['heads'][f'{held}_{seed}_motion'])
            out = control.matched(d, bank, menu, count, packet, p, cuts, ref)
            for arm in ('anchor', 'prefix_match', 'comoving_ego'):
                if out['policies'][arm] != past['policies'][arm]:
                    raise ValueError(f'Fixed-slot control changed: {held}/{seed}/{arm}')
            if out['oracle']['comoving_ego'] != past['oracle']['comoving_ego']:
                raise ValueError('Frozen EqMotion capacity did not replay')
            rows.append(dict(held=held, seed=seed, **out))
        print(dict(held=held, cells=len(rows)), flush=True)
    if len(rows) != 21:
        raise ValueError('Expected seven sites by three seeds')
    summaries = {k: {f: sum(r['policies'][k][f] for r in rows)
                  for f in ('edits', 'views', 'undefined', 'risk_pass', 'dual_positive')}
                 for k in rows[0]['policies']}
    if any((v['edits'], v['views']) != (1457, 606) for v in summaries.values()):
        raise ValueError('Recipient or recording budget changed')
    capacity = {k: dict(sum((Counter(r['oracle'][k]) for r in rows), Counter())) for k in rows[0]['oracle']}
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))
    effects = {}
    for k in rows[0]['effects_vs_anchor']:
        z = np.array([np.mean([r['effects_vs_anchor'][k] for r in rows if r['held'] == s]) for s in sites])
        effects[k] = dict(point=float(z.mean()), interval95=np.quantile(z[draw].mean(1), [.025, .975]).tolist())
    run.once(m.OUT/'slot_readout.json', run.stable(dict(registration=run.artifact(reg), policies=summaries,
        capacity=capacity, effects=effects, cells=rows, original_2pct=True, independent_confirmation=False,
        confirmation_access=False, deployment_changed=False)))
    print(dict(policies=summaries, capacity=capacity, effects=effects), flush=True)


if __name__ == '__main__':
    main()
