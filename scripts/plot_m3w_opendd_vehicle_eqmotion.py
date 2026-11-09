"""Plot the frozen vehicle forecast comparison without accessing any trajectory."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT/'outputs/publication_readiness_2026_09/opendd_vehicle_eqmotion_v1/readout.json'


def panel(ax, rows, title):
    colors = ('#15516b', '#a15529', '#475467')
    for i, ((name, record), color) in enumerate(zip(rows, colors, strict=True)):
        x = record['mean']
        lo, hi = record['ci95']
        ax.plot(record['sites'], [i]*len(record['sites']), 'o', color=color, alpha=.27,
                markersize=4, markeredgewidth=0)
        ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt='D', color=color,
                    markersize=4.5, capsize=3, linewidth=1.4)
    ax.axvline(0, color='#68717a', lw=.8, ls='--')
    ax.set_yticks(range(len(rows)), labels=[name for name, _ in rows])
    ax.set_ylim(len(rows)-.4, -.6)
    ax.set_xlabel('Conservative gain (pp of observed CV error)')
    ax.set_title(title, loc='left', fontsize=11, fontweight='semibold', pad=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#e0e3e7', linewidth=.6)
    ax.set_axisbelow(True)


def render(path, data):
    if data['independent_confirmation'] or data['external_outcomes_opened']:
        raise ValueError('This figure is labeled as OpenDD development only')
    summary, pairs = data['metrics'], data['paired']
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.3, 5.4))
    panel(ax, [
        ('EqMotion + neighbors', summary['comoving_peers']['cv_gain']['lower']),
        ('EqMotion ego only', summary['comoving_ego']['cv_gain']['lower']),
        ('Vehicle incumbent', summary['completion']['cv_gain']['lower']),
    ], '(a) Against constant velocity')
    panel(bx, [
        ('Neighbors vs incumbent', pairs['comoving_peers_minus_completion']['lower']),
        ('Ego only vs incumbent', pairs['comoving_ego_minus_completion']['lower']),
        ('Neighbors vs ego only', pairs['comoving_peers_minus_comoving_ego']['lower']),
    ], '(b) Direct paired comparisons')
    fig.subplots_adjust(left=.36, right=.97, top=.94, bottom=.17, hspace=.77)
    fig.text(.36, .055, 'Circles: seven location means. Diamonds: macro means.\n'
             'Bars: 95% paired location bootstrap intervals; three seeds per location.',
             fontsize=8.5, color='#39434c', va='bottom')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), metadata={'Title': 'OpenDD vehicle forecast control',
                'Author': '', 'CreationDate': None, 'ModDate': None})
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output, json.loads(READOUT.read_text()))
