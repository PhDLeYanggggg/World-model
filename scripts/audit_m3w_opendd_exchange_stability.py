"""Stress full-recording exchange witnesses without dropping either action."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_attrition as audit
from src.evaluation.m3w_selection_forecast_readout import errors


run, before, model, base, np, torch = audit.run, audit.before, audit.model, audit.base, audit.np, audit.torch
matched, parent = audit.matched, audit.matched.parent
capacity = parent.prior.prior.capacity
OUT = audit.OUT.parent/'opendd_exchange_stability_v1'
RISKS = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')
RESERVES = {'risk_slack_0_25pp': .0025, 'risk_slack_0_50pp': .005, 'risk_slack_1_00pp': .01}
FEATURES = ('full_rescue', *RESERVES, 'utility_0_10pp', 'material_and_slack', 'three_blocks_supported',
            'stable_risk', 'stable_dual', 'stable_material_and_slack', 'two_easy_support_queries')


def retained_queries(n, src, dst):
    protected = {src, dst}
    masks = []
    for block in np.array_split(np.arange(n), 3):
        removed = [i for i in block if i not in protected]
        if removed:
            mask = np.ones(n, dtype=bool)
            mask[removed] = False
            masks.append(mask)
    return masks


def risk_record(local, cv, prediction, cut):
    y, v, rec = (local[k] for k in ('y', 'valid', 'rec'))
    records = capacity.policy(prediction, cv, cv, y, v, rec, easy_cut=cut[0], hard_cut=cut[1])['recordings']
    if len(records) != 1:
        raise ValueError('Risk stress requires exactly one recording')
    return next(iter(records.values()))


def stress(local, cv, ols, anchor, swapped, cut, full, query, n, src, dst):
    r = risk_record(local, cv, swapped, cut)
    if not r['supported'] or not full['dual_positive']:
        raise ValueError('Only full-roster witnesses may enter the stress test')
    slack = .02 - max(r[k] for k in RISKS)
    gain = min(full['vs_cv'], full['vs_ols4'])
    masks = retained_queries(n, src, dst)
    y, valid = local['y'], local['valid']
    error = errors(cv, y, valid)[0]
    edited = np.linalg.norm(swapped-cv, axis=-1).max(1) > 0.
    easy = valid.any(1) & (error > 0.) & (error <= cut[0])
    good, counts = {}, Counter()
    for m in masks:
        take = m[query]
        if not take.any() or not take[query == src].all() or not take[query == dst].all():
            raise ValueError('A stress deletion removed part of an action')
        if (np.linalg.norm(anchor[take]-cv[take], axis=-1).max(1) > 0.).sum() != edited[take].sum():
            raise ValueError('Stress deletion broke matched edited-row counts')
        d = {k: v[take] for k, v in local.items()}
        p = swapped[take]
        if np.linalg.norm(cv[take]-d['y'], axis=-1)[d['valid']].sum() <= 0.:
            state = 'unknown_risk'
        else:
            state = audit.status(parent.score(d, cv[take], ols[take], p, cut))
        counts[state] += 1
    supported = len(masks) == 3
    good.update(full_rescue=True, **{k: slack >= v for k, v in RESERVES.items()},
                utility_0_10pp=gain > .10, material_and_slack=(gain > .10 and slack >= .005),
                three_blocks_supported=supported,
                stable_risk=supported and not (counts['unknown_risk'] or counts['violates_risk']),
                stable_dual=supported and counts['dual_pass'] == 3,
                stable_material_and_slack=supported and counts['dual_pass'] == 3 and gain > .10 and slack >= .005,
                two_easy_support_queries=len(np.unique(query[easy & edited])) >= 2)
    return good, counts


def evaluate(site, seed, motion_key, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(bank[0], path, rows)
    counts, records, candidates, states = Counter(), Counter(), Counter(), Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound anchor')
        local = {k: d[k][ai] for k in ('y', 'valid', 'rec')}
        query = np.searchsorted(np.flatnonzero(q), d['query'][ai])
        cv, ols = bank[0][ai], bank[1][ai]
        p0, score = before.measure(d, ai, bank, path, r, a, cut)
        pairs = model.exchange_menu(a, legal[q], move[q], cap)
        counts.update(views=1, edits=int(a.sum()), legal_exchanges=len(pairs), anchor_pass=score['dual_positive'])
        chosen = None
        if len(pairs):
            z = torch.from_numpy(model.features(x[q], legal[q], move[q], a, cap, pairs)[0])
            with torch.inference_mode():
                chosen = model.rank(nets['aligned'](matched.exp.design(z, width, 'aligned')))
        if chosen is None:
            counts.update(aligned_dual=score['dual_positive'])
        view = set()
        for j, (src, dst) in enumerate(pairs):
            p, s = before.measure(d, ai, bank, path, r, model.exchanged(a, src, dst), cut)
            if j == chosen:
                counts.update(chosen=1, aligned_dual=s['dual_positive'])
            if not s['dual_positive'] or score['dual_positive']:
                continue
            outcomes, stress_states = stress(local, cv, ols, p0, p, cut, s, query, len(a), src, dst)
            view.update(k for k, v in outcomes.items() if v)
            candidates.update(k for k, v in outcomes.items() if v)
            states.update(stress_states)
        records.update(view)
        if chosen is not None:
            counts.update(fixed_mask_rescue=int('full_rescue' in view), fixed_mask_stable=int('stable_dual' in view))
    return dict(counts=dict(counts), views_with_witness=dict(records), rescue_candidates=dict(candidates),
                deletion_outcomes=dict(states))


def summarize(groups):
    rows = [v for group in groups.values() for v in group]
    keys = ('counts', 'views_with_witness', 'rescue_candidates', 'deletion_outcomes')
    out = {k: dict(sum((Counter(r[k]) for r in rows), Counter())) for k in keys}
    for k in ('views_with_witness', 'rescue_candidates'):
        for f in FEATURES:
            out[k].setdefault(f, 0)
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    x = np.array([np.mean([100.*r['views_with_witness'].get('stable_dual', 0)/r['counts']['views']
                           for r in g]) for g in groups.values()])
    out['stable_rescue_rate_pp'] = dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    return out


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inputs = [Path(__file__), OUT/'protocol.md', audit.OUT/'readout.json', audit.OUT/'registration.json']
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in inputs], reserves=RESERVES,
        block_count=3, original_2pct=True, no_training=True, external_outcome_access=False)))
    groups = {}
    for args, ref in matched.context_cells():
        site, seed = args[:2]
        row = evaluate(*args)
        if row['counts'].get('chosen', 0) != ref['transitions']['aligned']['exchanged'] or (
                row['counts'].get('aligned_dual', 0) != ref['scores']['aligned']['dual_positive']):
            raise ValueError('Frozen switch mask did not reproduce')
        groups.setdefault(site, []).append(dict(seed=seed, **row))
        print(site, seed, 'complete', flush=True)
    s = summarize(groups)
    ref = run.read(audit.OUT/'readout.json')['summary']['counts']
    for a, b in (('views','views'), ('edits','edits'), ('legal_exchanges','candidates'), ('anchor_pass','anchor_pass'),
                 ('chosen','switched'), ('fixed_mask_rescue','rescue_inside_mask')):
        if s['counts'][a] != ref[b]:
            raise ValueError(f'Frozen exchange roster changed: {a}')
    if s['views_with_witness']['full_rescue'] != ref['rescue_inside_mask']+ref['rescue_outside_mask']:
        raise ValueError('The exact witness count did not reproduce')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        original_2pct=True, no_training=True, external_outcome_access=False, diagnostic_only=True)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
