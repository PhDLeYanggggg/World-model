"""Registered training-time comparisons, never checkpoint selection."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_trajectory as run
from scripts.report_m3w_european_aux_gradient import interval

CONTRASTS = {'true_vs_cost': ('cap_aux', 'cost_only'), 'true_vs_shuffled': ('cap_aux', 'shuffled_aux')}


def aggregate(rows, cfg):
    paired = defaultdict(dict); grad = defaultdict(list)
    for row in rows:
        ident = row['identity']; assignment = f"P{ident['producer']}__C{ident['controller']}"
        base = (ident['pair'], assignment, row['step'])
        for site, scores in row['localities'].items():
            for stratum, metrics in scores.items():
                if stratum == 'true_event': continue
                for metric in ('cost4', 'easy_harm'):
                    key = base+(stratum, metric, site, ident['seed'], ident['excluded_locality'])
                    assert ident['arm'] not in paired[key]
                    paired[key][ident['arm']] = metrics[metric]
        for batch in row['gradients']:
            for label, values in batch.items():
                for metric in ('cost4_shared', 'easy_harm_shared'):
                    grad[base+(ident['arm'], label, metric)].append(values[metric])
    contrasts = defaultdict(lambda: defaultdict(list)); missing = defaultdict(int)
    for key, arms in paired.items():
        assert set(arms) == set(cfg['arms'])
        for name, (candidate, reference) in CONTRASTS.items():
            result_key = key[:5]+(name,)
            if arms[candidate] is None or arms[reference] is None:
                missing[result_key] += 1
            else:
                contrasts[result_key][key[5]].append((key[6], key[7], arms[candidate], arms[reference]))
    comparisons = []
    for key in sorted(set(contrasts) | set(missing)):
        sites = contrasts[key]; local = []; support = {}
        for site, entries in sorted(sites.items()):
            seeds = sorted(set(r[0] for r in entries))
            complete = seeds == cfg['seeds'] and all(
                len([r for r in entries if r[0] == seed]) == 3 and
                len(set(r[1] for r in entries if r[0] == seed)) == 3 for seed in seeds)
            support[site] = dict(context_seed_pairs=len(entries), complete=bool(complete))
            if complete:
                per_seed = [np.mean([(a, b) for s, _, a, b in entries if s == seed], axis=0) for seed in seeds]
                local.append(np.mean(per_seed, axis=0))
        comparisons.append(dict(pair=key[0], assignment=key[1], step=key[2], stratum=key[3],
            metric=key[4], contrast=key[5], missing_context_seed_pairs=missing[key], locality_support=support,
            **interval(local, cfg)))
    gradient_rows = []
    for key, entries in sorted(grad.items()):
        cos = [r['cosine'] for r in entries if r['cosine'] is not None]
        ratios = [r['norm_ratio'] for r in entries if r['norm_ratio'] is not None]
        gradient_rows.append(dict(pair=key[0], assignment=key[1], step=key[2], arm=key[3], label=key[4],
            metric=key[5], dependent_batches=len(entries), estimable=len(cos), negative=sum(c < 0 for c in cos),
            cosine_median=float(np.median(cos)) if cos else None,
            norm_ratio_median=float(np.median(ratios)) if ratios else None))
    timelines = defaultdict(list)
    for r in comparisons:
        timelines[(r['pair'], r['assignment'], r['stratum'], r['metric'], r['contrast'])].append(r)
    onsets = []
    for key, values in sorted(timelines.items()):
        values.sort(key=lambda r: r['step'])
        negative = [r['step'] for r in values if r['status'] != 'not_estimable' and r['CI'][1] < 0]
        positive = [r['step'] for r in values if r['status'] != 'not_estimable' and r['CI'][0] > 0]
        onsets.append(dict(pair=key[0], assignment=key[1], stratum=key[2], metric=key[3], contrast=key[4],
            first_observed_negative_step=min(negative) if negative else None, negative_steps=negative,
            positive_steps=positive, exact_onset_or_causality_established=False))
    return dict(comparisons=comparisons, gradients=gradient_rows, timelines=onsets,
        snapshots=len(rows), scientific_success=False, checkpoint_selected=False,
        generalization_measured=False, independent_replication=False, deployment_changed=False)


def load_rows():
    path = run.PUBLIC/'training_freeze.json'; run.parent.risk.base.previous.require_committed(path)
    freeze = json.loads(path.read_text()); rows = []
    assert freeze['registration'] == run.artifact(run.PUBLIC/'registration_lock.json')
    for ref in freeze['receipts']:
        assert ref == run.artifact(ROOT/ref['path'])
        doc = json.loads((ROOT/ref['path']).read_text())
        run.validate_receipt(ROOT/ref['path'], doc['identity'])
        assert doc['numerical_state_exact']
        rows.extend(json.loads((ROOT/a['path']).read_text()) for a in doc['snapshots'])
    assert len(rows) == 2160
    return rows


def write_report(result):
    lines = ['# Training-Time Auxiliary Diagnostics', '',
        'fresh_run: 432 exact reconstructions, 864,000 optimizer updates and 2,160 intermediate measurements.',
        'cached_verified: unchanged training features, nested labels, initializations, samplers and original final states.',
        'not_run: new model variant, checkpoint selection, outer policy readout or independent evaluation.', '',
        'These are normalized fitting losses, not ADE/FDE gains. Positive percentages mean lower fitting loss.',
        'All final model, optimizer, sampling and random-number states match the original experiment exactly.',
        'Three seeds and three fitting contexts are averaged within each of four localities; 3,000 paired locality',
        'resamples give descriptive fitting intervals. These are not independent replication or confirmation.', '',
        '## Full-Input Positive-Envelope Easy-Harm Loss', '',
        '| Assignment | Contrast | Update | Reduction (%) | Descriptive 95% interval |', '|---|---|---:|---:|---|']
    for r in result['comparisons']:
        if r['pair'] == 'full' and r['stratum'] == 'envelope_positive' and r['metric'] == 'easy_harm':
            lines.append(f"| {r['assignment']} | {r['contrast']} | {r['step']} | {r.get('point')} | {r.get('CI')} |")
    lines += ['', 'Every family, assignment, severity stratum and gradient result is retained in aggregate_metrics.json.',
        'Stratum thresholds use fitting-only equal-locality weights. Empty or incomplete strata are not_estimable.',
        'First observed adverse intervals are coarse time diagnostics, not exact onset or causal attribution.',
        'The shared source loader is not a filesystem blind; fitting rows are selected before label indexing.',
        'Independent selection, reserved calibration and confirmation remain unopened.',
        'Obs8/pred12 native annotation steps in detector pixels. No seconds/metric, physical-safety, human-gold,',
        'true3D/foundation claim. Stage5C and SMC remain disabled. No deployment change.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'm3w_aux_trajectory_v1'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, pair in enumerate(('full', 'motion_only')):
        for col, contrast in enumerate(CONTRASTS):
            ax = axes[row, col]
            use = [r for r in result['comparisons'] if r['pair'] == pair and r['contrast'] == contrast
                   and r['metric'] == 'easy_harm' and r['stratum'] == 'envelope_positive']
            for assignment in sorted(set(r['assignment'] for r in use)):
                rr = sorted([r for r in use if r['assignment'] == assignment and r['status'] != 'not_estimable'], key=lambda r: r['step'])
                ax.plot([r['step'] for r in rr], [r['point'] for r in rr], marker='o', markersize=3, label=assignment)
            ax.axhline(0, color='black', linewidth=.8); ax.set_title(f'{pair}: {contrast}')
            ax.set_ylabel('Fitting HE loss reduction (%)'); ax.set_xlabel('Optimizer update')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Training diagnostics only; all terminal states exactly reconstructed')
    fig.tight_layout(); fig.savefig(run.PUBLIC/'training_time_effects.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'training_time_effects.png', dpi=160); plt.close(fig)


def main():
    cfg, _ = run.registration(); result = aggregate(load_rows(), cfg)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    write_report(result); plot(result)
    print(json.dumps(dict(snapshots=result['snapshots'], comparisons=len(result['comparisons']), gradients=len(result['gradients']))))


if __name__ == '__main__': main()
