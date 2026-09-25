"""Plot both fixed-count anchors without selecting a favorable fold or event."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.report_m3w_european_hurdle_coverage import verify, run


def main():
    a = verify()
    rows = [(fold, seed, event) for fold in range(3) for seed in (17, 29, 43) for event in ('all', 'easy')]
    colors = ('#146b78', '#ad4936')
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, layout='constrained')
    for ax, component, title in zip(axes, ('ranking_at_product_count', 'ranking_at_hurdle_count'),
            ('At original product-MSE common counts', 'At original hurdle common counts')):
        for i, (fold, seed, event) in enumerate(rows):
            for candidate, color, offset in zip(('neural', 'damping097'), colors, (-.14, .14)):
                key = f'{candidate}_fold{fold}_seed{seed}_{event}'
                value = a['decomposition'][key]['all'][component]
                if value is None:
                    continue
                y = i + offset
                ax.plot(value['ci95_pp'], [y, y], color=color, linewidth=.9)
                ax.plot(value['mean_gain_difference_pp'], y, 'o', color=color, markersize=4)
        ax.axvline(0, color='#777777', linewidth=.8)
        ax.grid(axis='x', color='#dddddd', linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Hurdle ranking advantage (pp of CV-normalized ADE)')
    axes[0].set_yticks(range(len(rows)), [f'F{f} / S{s} / {e}' for f, s, e in rows])
    axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=c, marker='o', linestyle='none', label=n)
                       for c, n in zip(colors, ('Neural candidate', 'Damping candidate'))],
               loc='outside lower center', ncols=2)
    fig.suptitle('Does occurrence/severity supervision improve ordering at the same count?\n'
                 'All 72 matched contrasts; conditional locality-bootstrap intervals. Offline diagnostics, not deployable rules.', fontsize=12)
    path = run.PUBLIC / 'matched_ranking.svg'
    fig.savefig(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    fig.savefig(run.PUBLIC / 'matched_ranking.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(figures=1, matched_contrasts=72)))


if __name__ == '__main__':
    main()
