"""Complete prespecified contrasts; fitting diagnostics, not held-out evidence."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_gradient as run

CONTRASTS = {'true_minus_cost': ('true_aux', 'cost_only'),
             'true_minus_shuffled': ('true_aux', 'shuffled_aux'),
             'projection_minus_true': ('projected_true', 'true_aux'),
             'projected_true_minus_shuffled': ('projected_true', 'projected_shuffled')}


def interval(values, cfg):
    array = np.asarray(values, float)
    if len(array) != 4 or not np.isfinite(array).all() or array[:, 1].mean() <= 0:
        return dict(status='not_estimable', localities=len(array))
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = rng.integers(0, len(array), (cfg['bootstrap_resamples'], len(array)))
    means = array[draws].mean(1)
    if np.any(means[:, 1] <= 0): return dict(status='not_estimable', localities=len(array))
    dist = 100*(means[:, 1]-means[:, 0])/means[:, 1]
    return dict(status='fitting_diagnostic_only', point=100*(array[:, 1].mean()-array[:, 0].mean())/array[:, 1].mean(),
                CI=np.quantile(dist, [.025, .975]).tolist(), localities=4,
                mean_candidate=float(array[:, 0].mean()), mean_reference=float(array[:, 1].mean()))


def aggregate(rows, cfg):
    cell = defaultdict(lambda: defaultdict(list)); grad = defaultdict(list); unsupported = 0
    for row in rows:
        assignment = f"P{row['producer']}__C{row['controller']}"
        for state in row['states']:
            key = (row['pair'], state['arm'], assignment)
            for repeat in state['repeats']:
                for label in ('true', 'shuffled'):
                    g = repeat['gradients'][label]
                    grad[key+(label,)].append(g)
                    unsupported += g['losses']['known_events'] == 0
                for contrast, (candidate, reference) in CONTRASTS.items():
                    a = repeat['after'][candidate]['localities']; b = repeat['after'][reference]['localities']
                    for site in a:
                        for metric in cfg['probe_metrics']:
                            av, bv = a[site][metric], b[site][metric]
                            if av is not None and bv is not None:
                                cell[key+(contrast, metric)][site].append((row['seed'], av, bv))
    comparisons = []
    for key, sites in sorted(cell.items()):
        local = []
        support = {}
        for site, entries in sorted(sites.items()):
            seeds = sorted(set(r[0] for r in entries)); assert seeds == [17, 29, 43]
            expected = 3*cfg['repeats']
            if any(sum(r[0] == seed for r in entries) != expected for seed in seeds):
                support[site] = 'incomplete_positive_probe_support'
                continue
            per_seed = [np.mean([(a, b) for s, a, b in entries if s == seed], axis=0) for seed in seeds]
            local.append(np.mean(per_seed, axis=0)); support[site] = len(entries)
        comparisons.append(dict(pair=key[0], arm=key[1], assignment=key[2], contrast=key[3],
            metric=key[4], locality_support=support, **interval(local, cfg)))
    gradients = []
    for key, values in sorted(grad.items()):
        for metric in ('cost4_shared', 'cost4_full', 'easy_harm_positive_shared', 'easy_harm_positive_full'):
            cosine = [r[metric]['cosine'] for r in values if r[metric]['cosine'] is not None]
            ratio = [r[metric]['norm_ratio'] for r in values if r[metric]['norm_ratio'] is not None]
            gradients.append(dict(pair=key[0], arm=key[1], assignment=key[2], label=key[3], metric=metric,
                batches=len(values), estimable=len(cosine), negative=sum(x < 0 for x in cosine),
                cosine_median=float(np.median(cosine)) if cosine else None,
                norm_ratio_median=float(np.median(ratio)) if ratio else None))
    screen = {}
    for contrast in ('projection_minus_true', 'projected_true_minus_shuffled'):
        for metric in ('cost4', 'easy_harm_positive'):
            chosen = [r for r in comparisons if r['pair'] == 'full' and r['arm'] == 'cap_aux'
                      and r['contrast'] == contrast and r['metric'] == metric]
            screen[contrast+'/'+metric] = (len(chosen) == 6 and
                all(r['status'] != 'not_estimable' and r['CI'][1] >= 0 for r in chosen) and
                sum(r.get('point', 0) > 0 for r in chosen) >= 5)
    return dict(comparisons=comparisons, gradients=gradients, unsupported_auxiliary_batch_uses=unsupported,
        repair_screen=screen, separate_projection_training_warranted=all(screen.values()),
        initial_shared_not_estimable=sum(r['initial']['shared']['cosine'] is None for r in rows),
        scientific_success=False, generalization_measured=False, deployment_changed=False)


def main():
    cfg, _ = run.registration()
    path = run.PUBLIC/'diagnostic_freeze.json'; run.parent.risk.base.previous.require_committed(path)
    freeze = json.loads(path.read_text()); rows = []
    for a in freeze['receipts']:
        assert run.artifact(ROOT/a['path']) == a
        row = json.loads((ROOT/a['path']).read_text())
        assert row['registration'] == run.artifact(run.PUBLIC/'registration_lock.json')
        rows.append(row)
    assert len(rows) == 144
    result = aggregate(rows, cfg)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    lines = ['# Fitting-Only Gradient and AdamW Results', '',
        'fresh_run: 17,280 isolated virtual AdamW updates from 432 frozen step-2000 states;',
        '3,456 final-state batch diagnostics and 144 initial checks. No new fully trained model.',
        'cached_verified: fitting features, nested targets, preprocessing, checkpoints and moments.',
        'not_run: intermediate trajectory checkpoints, new full training, held readout, new policy,',
        'independent selection/calibration/confirmation. No deployment change.', '',
        'Positive percentage means lower actual fitting-probe loss after the isolated step.',
        'Not trajectory improvement, held-out performance, physical safety or a causal account of training failure.',
        'Probe rows are disjoint from the single update, but all are previously trained on.', '',
        '## Primary Repair Screen', '```json', json.dumps(result['repair_screen'], indent=2), '```',
        f"Separate projection training warranted: {result['separate_projection_training_warranted']}", '',
        '## Full-Input Cap-Auxiliary Frozen States', '',
        '| Assignment | Contrast | Metric | Point (%) | 95% descriptive locality interval |', '|---|---|---|---:|---|']
    for r in result['comparisons']:
        if r['pair'] == 'full' and r['arm'] == 'cap_aux':
            lines.append(f"| {r['assignment']} | {r['contrast']} | {r['metric']} | {r.get('point')} | {r.get('CI')} |")
    lines += ['', 'All assignments, full/motion families, three frozen arms and gradients are in aggregate_metrics.json.',
        'Locality averages combine eight repeats, three fitting contexts and three seeds; four localities are',
        'resampled 3000 times. Models and assignments share fitting data. These intervals are descriptive,',
        'not independent confirmation or a generalization guarantee. No window-level inferential sample size.',
        'Projection protects the raw shared four-cost gradient only. AdamW moments, clipping and finite steps',
        'can change actual effects, and easy-harm is not individually protected. Null gradients are unsupported.',
        'End-state diagnostics cannot establish early/mid-training behavior. Known zero-envelope costs remain included.',
        'Obs8/pred12 annotation steps, detector pixels. No metric/seconds, physical-safety, human-gold,',
        'true3D/foundation claims, Stage5C execution or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'gradients')}, indent=2))


if __name__ == '__main__': main()
