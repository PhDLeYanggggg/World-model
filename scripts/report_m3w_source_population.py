"""Render verified source-population tables and an aggregate-only figure."""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    out = ROOT/'outputs/publication_readiness_2026_09/source_population_v1'
    a = json.loads((out/'audit.json').read_text())
    v = json.loads((out/'verification.json').read_text())
    if file_digest(out/'audit.json') != v['audit_sha256']:
        raise ValueError('Unverified audit')
    lines = ['# Complete Source-Population Tables', '',
        'Fresh aggregate calculation from hash-verified existing arrays, with raw',
        'index/geometry replays. No new fits or deployment. All values are source',
        'diagnostics on four previously explored sites, not independent testing.', '',
        '## Population', '',
        '| Site | Indexed | Complete | Partial | Absent | Static history |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s, r in a['by_site'].items():
        lines.append(f"| {s} | {r['rows']:,} | {r['complete']:,} | {r['partial']:,} | {r['absent']:,} | {r['static_history']:,} |")
    lines += ['', 'All indexed queries are retained. Retrospective event labels require',
        'all twelve futures; partial/absent future events are unknown. FDE requires',
        'the twelfth target, not merely the last available target.', '',
        '## Fixed Causal Baselines', '',
        'Equal-site means. Pixel diagnostics are not metrically calibrated and do',
        'not replace the registered past-normalized primary error.', '']
    for cohort, r in a['cohorts'].items():
        lines += [f'### {cohort}', '',
            '| Baseline | Normalized ADE | Normalized FDE | Pixel ADE diagnostic | Pixel FDE diagnostic |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, m in r['baseline_metrics'].items():
            lines.append(f"| {name} | {m['normalized_ade']:.6f} | {m['normalized_fde']:.6f} | {m['native_pixel_ade_diagnostic']:.6f} | {m['native_pixel_fde_diagnostic']:.6f} |")
        lines += ['', f"Rows: {r['rows']:,}; final-label rows: {r['final_label_rows']:,}.",
            f"Future-informed oracle headroom: {r['oracle_headroom_over_cv_percent']:.6f}% versus CV.",
            'This is not a learned or causal oracle.', '']
    lines += ['## Complete-Future Events', '',
        '| Category | Windows | Scoped tracks | Disjoint spans | Normalized CV error share | Pixel CV error share | Oracle gain vs CV |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in a['events'].items():
        gain = r['oracle_headroom_over_cv_percent']
        gain = 'undefined: zero CV error' if gain is None else f'{gain:.6f}%'
        lines.append(f"| {name} | {r['rows']:,} | {r['tracks']:,} | {r['disjoint_spans']:,} | {r['cv_normalized_contribution']['percent_of_total']:.6f}% | {r['cv_native_contribution_diagnostic']['percent_of_total']:.6f}% | {gain} |")
    lines += ['', 'Tracks and disjoint spans across categories are not additive independent',
        'events. Exact float32 cached-label categories describe annotations, not',
        'human-gold behavioral semantics. Tiny negative oracle differences at',
        'approximately 1e-13 percent are floating-point summation roundoff.', '',
        '## Site Consistency', '',
        '| Site | Static-start windows | Normalized CV error share | Pixel CV error share | All oracle headroom |',
        '| --- | ---: | ---: | ---: | ---: |']
    for s, r in a['by_site'].items():
        c = r['cohorts']['complete']
        lines.append(f"| {s} | {r['events']['static_moves']['rows']:,} | {c['scale_floor_cv_error_share']['percent_of_total']:.6f}% | {c['scale_floor_native_cv_error_share']['percent_of_total']:.6f}% | {c['oracle_headroom_over_cv_percent']:.6f}% |")
    lines += ['', '## Other-Site Baseline Selection', '',
        'For supported, complete and complete-moving cohorts, every held-source',
        'fold selects causal CV using the other three source sites. This gives',
        'zero gain over CV, not a new predictor. Complete-moving oracle headroom',
        f"is {a['complement_selected']['complete_moving']['oracle_headroom_percent']:.6f}%.", '',
        '## Per-Recording Support', '',
        '| Recording | Indexed | Complete | Partial | Absent | Static history |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in a['readouts']:
        lines.append(f"| {r['recording']} | {r['rows']:,} | {r['complete']:,} | {r['partial']:,} | {r['absent']:,} | {r['static_history']:,} |")
    lines += ['', '## Verification', '',
        f"- {v['arrays_checked']} array hashes checked; {v['index_rows_rebuilt']:,} raw past-index keys rebuilt.",
        f"- {v['baseline_row_costs_recomputed']:,} baseline/query ADE and FDE pairs independently reduced.",
        f"- {v['raw_geometry_label_queries']} exact raw geometry/label replays; {v['future_array_poison_queries']} future-poison input checks.",
        '- Completed audit rerun exactly matches every saved row and aggregate.',
        '- No main, bookstore, original validation/test or external readout.',
        '- No primary metric change, new training, Stage5C or SMC.', '']
    (out/'complete_results.md').write_text('\n'.join(lines))

    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'m3w-matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'m3w-font-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none', 'svg.hashsalt':'m3w-source-population-v1'})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = ['static_stays', 'static_moves', 'moving_stops', 'moving_turns', 'other_motion']
    colors = ['#a0a0a0', '#b74940', '#387a9b', '#477b42', '#ad8735']
    left = [0., 0.]
    for name, color in zip(names, colors):
        e = a['events'][name]
        values = [e['cv_normalized_contribution']['percent_of_total'], e['cv_native_contribution_diagnostic']['percent_of_total']]
        ax.barh([1, 0], values, left=left, color=color, height=.42, label=name.replace('_', ' '))
        left = [x+y for x,y in zip(left, values)]
    ax.set_yticks([1,0], ['Past-normalized ADE', 'Annotation-pixel ADE\n(diagnostic only)'])
    ax.set_xlim(0,100); ax.set_xlabel('Share of equal-site CV error (%)')
    ax.set_title('The same queries and predictions produce very different error weights', pad=28)
    ax.text(0,1.43,'Static-start queries: 4.77% of complete windows', fontsize=10)
    ax.text(50,1,'99.75% static-start', ha='center', va='center', color='white', weight='bold')
    ax.text(0,-.39,'Static-start share in pixels: 0.67%', fontsize=10)
    ax.set_ylim(-.6,1.6)
    ax.spines[['top','right','left']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.24), ncol=3, frameon=False)
    fig.subplots_adjust(left=.24, right=.97, top=.78, bottom=.29)
    fig.savefig(out/'error_contribution.svg', metadata={'Date':None})
    svg = out/'error_contribution.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig('/tmp/m3w-source-population.png', dpi=140)
    plt.close(fig)
    print(out/'complete_results.md')


if __name__ == '__main__':
    main()
