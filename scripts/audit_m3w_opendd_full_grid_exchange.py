"""Re-evaluate exact one-exchange headroom under full-grid risk."""

from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_attrition as original
from src.evaluation import m3w_full_grid_risk as risk

run, before, model, base, np, torch = original.run, original.before, original.model, original.base, original.np, original.torch
matched = original.matched
OUT = original.OUT.with_name('opendd_full_grid_exchange_v1')
PRIVATE = before.PRIVATE.with_name(OUT.name)
ARMS = ('incumbent', 'aligned', 'uniform', 'mask_oracle', 'menu_or_keep')
STATES = ('all_risk', 'easy_support', 'easy_excess')
LIMIT = .02
MATERIAL_PP = .1


def certificate(cv, proposal, target, valid, cut, score):
    """Return whether an original dual pass survives and list failure modes."""
    if not score['dual_positive']:
        return False, ()
    take = (proposal != cv).any(axis=(1, 2))
    if int(take.sum()) != score['edits'] or not take.any() or min(score['vs_cv'], score['vs_ols4']) <= 0.:
        raise ValueError('Original dual-pass inputs or roster changed')
    args = tuple(x[take] for x in (cv, proposal, target, valid))
    all_risk = risk.assess(*args)
    easy = risk.assess_easy(*args, easy_cut=cut, limit=LIMIT)
    full = all_risk['full_grid_upper']
    states = (full is None or full > LIMIT, not easy['easy_support_guaranteed'], easy['easy_excess_supremum'] > 0.)
    if all_risk['partial_tracks'] == 0 and all_risk['unknown_tracks'] == 0 and any(states):
        raise ValueError('Complete edited paths cannot lose an original dual pass')
    return not any(states), tuple(k for k, failed in zip(STATES, states, strict=True) if failed)


def summarize_view(choices, index):
    """Anchor is choice zero; other choices are legal exchanges only."""
    if not choices or (index is not None and not 1 <= index < len(choices)):
        raise ValueError('Invalid frozen exchange decision')
    z = np.asarray(choices, dtype=float)
    incumbent, selected = z[0], z[index or 0]
    uniform = incumbent if index is None else z[1:].mean(0)
    forced = incumbent if index is None else z[1:].max(0)
    return dict(zip(ARMS, (incumbent, selected, uniform, forced, z.max(0)), strict=True))


def evaluate(site, seed, motion_key, d, bank, menu, width, nets, cut):
    x, legal, rows, path, _ = menu
    move = base.model.movement_prefix(bank[0], path, rows)
    counts, losses = Counter(), Counter()
    outcomes = {k: np.zeros(3) for k in ARMS}
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = before.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        plan, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion_key, site=site, seed=seed, ordinal=i) or plan.sum() != f['edits']:
            raise ValueError('Unbound incumbent or edit count')
        pairs = model.exchange_menu(plan, legal[q], move[q], cap)
        index = None
        if len(pairs):
            feat, _ = model.features(x[q], legal[q], move[q], plan, cap, pairs)
            with torch.inference_mode():
                pick = model.rank(nets['aligned'](matched.exp.design(torch.from_numpy(feat), width, 'aligned')))
            if pick is not None:
                index = pick+1
        counts.update(views=1, edits=int(plan.sum()), candidates=len(pairs), switched=int(index is not None))
        cv, y, valid = bank[0][ai], d['y'][ai], d['valid'][ai]
        choices = []

        def compare(choice):
            p, score = before.measure(d, ai, bank, path, r, choice, cut)
            full, blocked = certificate(cv, p, y, valid, cut[0], score)
            if blocked:
                losses.update(blocked)
                losses['any'] += 1
            counts.update(original_passing_options=int(score['dual_positive']), full_grid_passing_options=int(full))
            return (score['dual_positive'], full, full and min(score['vs_cv'], score['vs_ols4']) > MATERIAL_PP)

        choices.append(compare(plan))
        for src, dst in pairs:
            choices.append(compare(model.exchanged(plan, src, dst)))
        for arm, value in summarize_view(choices, index).items():
            outcomes[arm] += value
    return dict(counts=dict(counts), outcomes={k: dict(zip(('original', 'full_grid', 'material_full_grid'),
                v.tolist(), strict=True)) for k, v in outcomes.items()}, losses=dict(losses))


def summary(groups, old):
    values = [r for g in groups.values() for r in g]
    counts = sum((Counter(r['counts']) for r in values), Counter())
    losses = sum((Counter(r['losses']) for r in values), Counter())
    results = {k: dict(sum((Counter(r['outcomes'][k]) for r in values), Counter())) for k in ARMS}
    prev = old['counts']
    if (any(counts[k] != prev[k] for k in ('views', 'edits', 'candidates', 'switched'))
            or results['incumbent']['original'] != prev['anchor_pass']
            or results['mask_oracle']['original'] != prev['same_mask_oracle_pass']
            or results['menu_or_keep']['original'] != prev['menu_or_keep_pass']
            or results['aligned']['original'] != old['arms']['aligned']['dual_pass']
            or not np.isclose(results['uniform']['original'], old['arms']['random']['dual_pass'], rtol=0., atol=1e-12)
            or counts['original_passing_options'] != counts['full_grid_passing_options']+losses['any']):
        raise ValueError('The original frontier and model results did not reproduce')
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    for left, right in (('menu_or_keep', 'incumbent'), ('menu_or_keep', 'aligned'), ('aligned', 'uniform')):
        rates = np.array([np.mean([100*(r['outcomes'][left]['full_grid']-r['outcomes'][right]['full_grid'])/
                                  r['counts']['views'] for r in g]) for g in groups.values()])
        contrasts[f'{left}_minus_{right}'] = dict(point_pp=float(rates.mean()),
            ci95_pp=np.quantile(rates[draws].mean(1), [.025, .975]).tolist())
    return dict(counts=dict(counts), losses=dict(losses), arms=results, full_grid_contrasts=contrasts,
                locations=len(groups), draws=len(draws))


def run_audit():
    files = [Path(__file__), Path(risk.__file__), OUT/'protocol.md', original.OUT/'readout.json',
             original.OUT/'registration.json', matched.exp.OUT/'head_freeze.json', matched.exp.OUT/'readout.json']
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], original_2pct=True,
        material_pp=MATERIAL_PP, no_training=True, new_external_outcome_access=False)))
    ident = run.sha(p)
    prior = run.read(original.OUT/'readout.json')['summary']
    groups = {}
    start = time.monotonic()
    for inputs, ref in matched.context_cells():
        site, seed = inputs[:2]
        path = PRIVATE/f'{site}_{seed}.json'
        if not path.exists():
            run.once(path, run.stable(dict(registration=ident, metrics=evaluate(*inputs))))
        item = run.read(path)
        if item['registration'] != ident:
            raise ValueError('Local checkpoint does not belong to this exact audit')
        r = item['metrics']
        if (r['counts']['switched'] != ref['transitions']['aligned']['exchanged']
                or r['outcomes']['aligned']['original'] != ref['scores']['aligned']['dual_positive']):
            raise ValueError('Original aligned mask or outcomes changed')
        groups.setdefault(site, []).append(dict(seed=seed, **r))
        event = dict(pid=os.getpid(), completed=sum(map(len, groups.values())), total=21,
                     site=site, seed=seed, seconds=time.monotonic()-start)
        with (PRIVATE/'heartbeat.pending').open('w') as f:
            json.dump(event, f)
        (PRIVATE/'heartbeat.pending').replace(PRIVATE/'heartbeat.json')
        print(event, flush=True)
    if len(groups) != 7 or any({r['seed'] for r in g} != {17, 29, 43} for g in groups.values()):
        raise ValueError('Missing frozen location or training seed')
    s = summary(groups, prior)
    run.once(OUT/'readout.json', run.stable(dict(registration=ident, summary=s, by_location=groups,
        result_source='fresh_retrospective_on_frozen_exposed_development', original_2pct=True, no_training=True,
        deployment_changed=False, new_external_outcome_access=False, independent_confirmation=False)))
    print(s, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_audit()
