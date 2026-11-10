"""Draw the OpenDD selection-gap figure from the registered aggregate readout."""

from io import BytesIO
import json
from pathlib import Path
from statistics import mean
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.m3w_reader_supplement_exchange.verify import check_exchange

READOUT = ROOT/'outputs/publication_readiness_2026_09/opendd_full_grid_exchange_v1/readout.json'
OUT = ROOT/'data/stage_cvpr2027_experiments/opendd_full_grid_exchange_v1/exchange_gap.pdf'
KEYS = ('incumbent', 'uniform', 'aligned', 'mask_oracle', 'menu_or_keep')
LABELS = ('Keep incumbent', 'Same-mask uniform', 'Aligned ranker', 'Same-mask hindsight', 'Any-view hindsight')
CONTRASTS = (('menu_or_keep', 'incumbent', 'Any-view hindsight vs. keep'),
             ('aligned', 'uniform', 'Aligned vs. same-mask uniform'))


def location_effects(data, left, right):
    return [mean(100*(s['outcomes'][left]['full_grid']-s['outcomes'][right]['full_grid'])/s['counts']['views']
                 for s in group) for group in data['by_location'].values()]


def render(data):
    a = check_exchange(data['summary'])
    if len(data['by_location']) != 7 or data['independent_confirmation']:
        raise ValueError('Expected the exposed seven-location result')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
                         'axes.labelsize': 8.5, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False})
    color, muted, dark = '#137978', '#79858D', '#202B33'
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 2.55), gridspec_kw={'width_ratios': [1, .85]})
    fig.subplots_adjust(left=.21, right=.97, top=.84, bottom=.20, wspace=.31)
    ax.set_title('(a) Same candidate and movement budgets', loc='left', weight='bold', pad=8)
    for i, k in enumerate(KEYS):
        original, robust, _ = a[k]
        y = len(KEYS)-i-1
        ax.hlines(y, robust, original, color=muted, linewidth=1.1)
        ax.scatter(original, y, marker='o', facecolors='white', edgecolors=muted, linewidth=1., s=30, zorder=3)
        ax.scatter(robust, y, marker='o', color=color, s=20, zorder=4)
        ax.annotate(f'{robust:.2f}' if k == 'uniform' else f'{robust:.0f}', (robust, y), xytext=(-5, 0),
                    textcoords='offset points', va='center', ha='right', color=dark, fontsize=8)
    ax.set_yticks(range(4, -1, -1), LABELS)
    ax.set(xlim=(0, 200), ylim=(-.85, 4.55), xlabel='Passing recording-seed views (of 606)', xticks=(0, 50, 100, 150, 200))
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#DDE2E5', linewidth=.5)
    ax.set_axisbelow(True)
    ax.scatter([], [], facecolors='white', edgecolors=muted, label='Original dual pass', s=25)
    ax.scatter([], [], color=color, label='Also full-grid', s=20)
    ax.legend(frameon=False, loc='upper right', fontsize=7.2, borderaxespad=.15, handletextpad=.25)

    bx.set_title('(b) Paired effects by location', loc='left', weight='bold', pad=8)
    bx.axvline(0, color=muted, linewidth=.7, linestyle='--')
    bx.set(xlim=(-6, 34), ylim=(-.2, 1.62), xlabel='Change in full-grid pass rate (pp)', xticks=(0, 10, 20, 30), yticks=[])
    for i, (l, r, label) in enumerate(CONTRASTS):
        y = 1-i*.77
        x = sorted(location_effects(data, l, r))
        row = data['summary']['full_grid_contrasts'][f'{l}_minus_{r}']
        p, (lo, hi) = row['point_pp'], row['ci95_pp']
        if abs(mean(x)-p) > 1e-10:
            raise ValueError('Plot inputs do not reproduce the paired point estimate')
        bx.text(-5.4, y+.36, label, va='center', fontsize=8, color=dark)
        bx.scatter(x, [y+.15+(j % 3)*.047 for j in range(len(x))], s=12, color=muted, zorder=3)
        bx.errorbar(p, y, xerr=[[p-lo], [hi-p]], color=color, marker='D', markersize=4,
                    linewidth=1.1, capsize=3, zorder=4)
    buf = BytesIO()
    fig.savefig(buf, format='pdf', metadata={'CreationDate': None, 'ModDate': None,
                                             'Title': 'OpenDD exchange headroom'})
    plt.close(fig)
    return buf.getvalue()


if __name__ == '__main__':
    raw = render(json.loads(READOUT.read_bytes()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_bytes() != raw:
        raise FileExistsError('Refusing to replace a frozen figure')
    if not OUT.exists():
        with OUT.open('xb') as f:
            f.write(raw)
    print(OUT)
