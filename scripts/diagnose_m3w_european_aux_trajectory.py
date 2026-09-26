"""Additive severity accounting and prespecified gradient-over-time displays."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import report_m3w_european_aux_trajectory as report
run = report.run
BINS = ('zero_easy_harm', 'severity_0_50', 'severity_50_90', 'severity_90_99', 'severity_99_100')


def diagnose(rows):
    paired = defaultdict(dict); bce = defaultdict(list)
    for r in rows:
        i = r['identity']; assign = f"P{i['producer']}__C{i['controller']}"
        for site, value in r['localities'].items():
            key = (i['pair'], assign, r['step'], site, i['seed'], i['excluded_locality'])
            assert i['arm'] not in paired[key]; paired[key][i['arm']] = value
            bce[key[:4]+(i['arm'],)].append(value['true_event']['BCE'])
    cell = defaultdict(lambda: defaultdict(list))
    for key, arms in paired.items():
        for name, (candidate, reference) in report.CONTRASTS.items():
            a, b = arms[candidate], arms[reference]
            n = b['envelope_positive']['rows']; assert a['envelope_positive']['rows'] == n and n > 0
            delta = [(b[k]['easy_harm_SSE']-a[k]['easy_harm_SSE'])/n for k in BINS]
            total = b['envelope_positive']['easy_harm']-a['envelope_positive']['easy_harm']
            np.testing.assert_allclose(sum(delta), total, rtol=1e-9, atol=1e-10)
            cell[key[:3]+(name,)][key[3]].append((key[4], key[5], *delta,
                a['envelope_positive']['easy_harm'], b['envelope_positive']['easy_harm']))
    contributions = []
    for key, sites in sorted(cell.items()):
        means = []
        for site, values in sorted(sites.items()):
            assert len(values) == 9 and len(set((v[0], v[1]) for v in values)) == 9
            assert sorted(set(v[0] for v in values)) == [17, 29, 43]
            means.append(np.mean([v[2:] for v in values], axis=0))
        assert len(means) == 4
        avg = np.mean(means, axis=0); baseline = avg[-1]
        pp = avg[:len(BINS)]/baseline*100
        contributions.append(dict(pair=key[0], assignment=key[1], step=key[2], contrast=key[3],
            additive_reduction_percentage_points=dict(zip(BINS, map(float, pp))),
            total_reduction_percent=float(100*(avg[-1]-avg[-2])/avg[-1]),
            equal_locality_context_seed_weighting=True))
    probabilities = defaultdict(list)
    for key, values in bce.items():
        assert len(values) == 9
        if any(v is None for v in values): continue
        probabilities[key[:3]+(key[4],)].append(float(np.mean(values)))
    p = [dict(pair=k[0], assignment=k[1], step=k[2], arm=k[3],
              fitting_localities=len(v), mean_true_event_BCE=float(np.mean(v))) for k, v in sorted(probabilities.items())]
    return dict(severity_contributions=contributions, true_event_BCE=p,
        interpretation='descriptive_fitting_error_accounting_not_causal_attribution', new_selection=False)


def plot(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'm3w_aux_trajectory_gradients_v1'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, pair in enumerate(('full', 'motion_only')):
        for col, metric in enumerate(('cost4_shared', 'easy_harm_shared')):
            ax = axes[row, col]
            use = [r for r in aggregate['gradients'] if r['pair'] == pair and r['arm'] == 'cap_aux'
                   and r['label'] == 'actual_training' and r['metric'] == metric]
            for assignment in sorted(set(r['assignment'] for r in use)):
                rr = sorted([r for r in use if r['assignment'] == assignment and r['estimable']], key=lambda r: r['step'])
                ax.plot([r['step'] for r in rr], [100*r['negative']/r['estimable'] for r in rr],
                        marker='o', markersize=3, label=assignment)
            ax.set_title(f'{pair}: {metric}'); ax.set_ylabel('Conflicting dependent batches (%)')
            ax.set_xlabel('Optimizer update'); ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Shared auxiliary conflicts over training; descriptive, not causal')
    fig.tight_layout(); fig.savefig(run.PUBLIC/'gradient_time_effects.svg', metadata={'Date': None})
    fig.savefig(run.PRIVATE/'gradient_time_effects.png', dpi=160); plt.close(fig)


def main():
    run.registration(); result = diagnose(report.load_rows())
    run.immutable_json(run.PUBLIC/'secondary_diagnosis.json', result)
    plot(json.loads((run.PUBLIC/'aggregate_metrics.json').read_text()))
    print(json.dumps(dict(severity_cells=len(result['severity_contributions']), BCE_cells=len(result['true_event_BCE']))))


if __name__ == '__main__': main()
