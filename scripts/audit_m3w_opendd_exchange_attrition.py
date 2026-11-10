"""Trace where the exact one-exchange opportunity is lost on OpenDD."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_context_matched as matched


run, before, model, base, np, torch = (matched.run, matched.before, matched.model, matched.base, matched.np, matched.torch)
OUT = matched.exp.OUT.parent/'opendd_exchange_attrition_v1'
CHOICES = ('aligned', 'gain_only', 'least_motion', 'random')
RISKS = ('unknown_risk', 'violates_risk', 'no_dual_utility', 'dual_pass')
BLOCKS = ('no_predicted_gain', 'below_transition_margin')


def status(score):
    if score['undefined']:
        return 'unknown_risk'
    if not score['risk_pass']:
        return 'violates_risk'
    return 'dual_pass' if score['dual_positive'] else 'no_dual_utility'


def trace(anchor, options, selections, block):
    """Partition exact recoveries; missing-risk cases never count as passes."""
    if bool(selections) != (block is None and bool(options)):
        raise ValueError('Inconsistent pre-outcome intervention')
    opts = [status(s) for s in options]
    pre = status(anchor)
    feasible = 'dual_pass' in opts
    choose = bool(selections)
    x = Counter(views=1, edits=anchor['edits'], candidates=len(opts), switched=int(choose))
    anchor_pass = pre == 'dual_pass'
    x['anchor_pass'] = int(anchor_pass)
    x['menu_or_keep_pass'] = int(anchor_pass or feasible)
    x['same_mask_oracle_pass'] = int(feasible if choose else anchor_pass)
    if not anchor_pass and feasible:
        group = 'outside_mask' if not choose else 'inside_mask'
        x[f'rescue_{group}'] += 1
        x[f'rescue_from_{pre}'] += 1
        if not choose:
            x[f'rescue_{block}'] += 1
    if anchor_pass and choose and not feasible:
        x['forced_unavoidable_loss'] += 1
    if choose:
        chosen = opts[selections['aligned']]
        if chosen != 'dual_pass':
            if anchor_pass:
                x[f'destroy_to_{chosen}'] += 1
            elif feasible:
                x[f'missed_in_mask_{chosen}'] += 1
    arms = {}
    for k in CHOICES:
        c = Counter({r: 0 for r in RISKS})
        if not choose:
            c[pre] += 1
        elif k == 'random':
            c.update({r: n/len(opts) for r, n in Counter(opts).items()})
        else:
            c[opts[selections[k]]] += 1
        c['rescued'] = c['dual_pass'] if not anchor_pass else 0
        c['destroyed'] = 1-c['dual_pass'] if anchor_pass else 0
        arms[k] = c
    return x, arms


def evaluate(site, seed, motion_key, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(bank[0], path, rows)
    counts, arms = Counter(), {k: Counter() for k in CHOICES}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound anchor or edit count')
        pairs = model.exchange_menu(a, legal[q], move[q], cap)
        take, block = {}, None
        if len(pairs):
            z, delta = model.features(x[q], legal[q], move[q], a, cap, pairs)
            with torch.inference_mode():
                scores = nets['aligned'](matched.exp.design(torch.from_numpy(z), width, 'aligned'))
                choice = model.rank(scores)
            if choice is None:
                block = BLOCKS[int(bool((scores[:, 3] > 0.).any()))]
            else:
                take = dict(aligned=choice, gain_only=int(scores[:, 3].argmax()), least_motion=int(delta.argmin()))
        # The four choices and fallback reason are fixed before scoring options.
        anchor = before.measure(d, ai, bank, path, r, a, cut)[1]
        opts = [before.measure(d, ai, bank, path, r, model.exchanged(a, src, dst), cut)[1] for src, dst in pairs]
        c, s = trace(anchor, opts, take, block)
        counts.update(c)
        for k in CHOICES:
            arms[k].update(s[k])
    return dict(counts=dict(counts), arms={k: dict(v) for k, v in arms.items()})


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    counts = sum((Counter(r['counts']) for r in rows), Counter())
    arms = {k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in CHOICES}
    if counts['menu_or_keep_pass'] != counts['anchor_pass']+counts['rescue_outside_mask']+counts['rescue_inside_mask']:
        raise ValueError('Exact rescue partition does not reconcile')
    if counts['same_mask_oracle_pass'] != counts['anchor_pass']+counts['rescue_inside_mask']-counts['forced_unavoidable_loss']:
        raise ValueError('Fixed-mask frontier does not reconcile')
    if counts['rescue_outside_mask'] != sum(counts[f'rescue_{r}'] for r in BLOCKS):
        raise ValueError('Missed guard reasons do not reconcile')
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(k):
        per_location = np.asarray([np.mean([100.*(r['arms'][k].get('dual_pass', 0)
            - r['arms']['random'].get('dual_pass', 0))/r['counts']['views'] for r in g]) for g in groups.values()])
        return dict(point_pp=float(per_location.mean()),
                    ci95_pp=np.quantile(per_location[draws].mean(1), [.025, .975]).tolist())

    return dict(counts=dict(counts), arms=arms, dual_rate_minus_random={k: ci(k) for k in CHOICES[:-1]})


def audit():
    files = [OUT/'protocol.md', Path(__file__), Path(matched.__file__),
             matched.exp.OUT/'registration.json', matched.exp.OUT/'readout.json',
             matched.exp.OUT/'head_freeze.json', matched.exp.OUT/'matched_swap_readout.json',
             before.OUT/'ranking_diagnostic.json']
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], controls=CHOICES,
        no_training=True, external_outcome_access=False, original_2pct=True)))
    groups = {}
    for inputs, ref in matched.context_cells():
        site, seed = inputs[:2]
        r = evaluate(*inputs)
        a, t = r['arms']['aligned'], ref['transitions']['aligned']
        if (r['counts']['switched'], a.get('rescued', 0), a.get('destroyed', 0), a.get('dual_pass', 0)) != (
                t['exchanged'], t['rescued'], t['destroyed'], ref['scores']['aligned']['dual_positive']):
            raise ValueError('Aligned mask or outcomes did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **r))
        print(f'{site}/{seed} complete', flush=True)
    s = summarize(groups)
    c = s['counts']
    frontier = run.read(before.OUT/'ranking_diagnostic.json')['summary']
    ref = run.read(matched.exp.OUT/'matched_swap_readout.json')['summary']
    if (c['views'], c['edits'], c['candidates'], c['anchor_pass'], c['menu_or_keep_pass'], c['switched']) != (
            frontier['views'], frontier['edits'], frontier['candidates'], frontier['anchor_dual'],
            frontier['exact_frontier']['positive'], ref['exchanged']):
        raise ValueError('Exchange diagnostic did not reproduce the frozen roster or frontier')
    if not np.isclose(s['arms']['random']['dual_pass'], ref['arms']['random']['dual_positive'], atol=1e-12, rtol=0.):
        raise ValueError('Exact chance control changed')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        no_training=True, original_2pct=True, external_outcome_access=False, offline_development=True)))
    print(s, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    audit()
