"""Render a two-panel capacity figure from the frozen scalar readout."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('readout', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    s = json.loads(args.readout.read_text())['summary']
    if s['bounds'] != ['menu', 'risk_relaxed', 'arbitrary'] or s['references'] != ['cv', 'ols4']:
        raise ValueError('Unknown frozen readout schema')
    x = np.arange(len(s['caps']))
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.1, 3.5), constrained_layout=True)
    for j, (name, color, marker) in enumerate((('Analytic menu, no risk test', '#35648d', 'o'),
           ('Analytic menu, relaxed risk', '#8b2d25', 's'), ('Arbitrary paths, no risk test', '#35714e', '^'))):
        a.plot(x, s['point_pp'][j][1], marker=marker, label=name, color=color, linewidth=1.65, markersize=4)
        b.plot(x, s['positive_records'][j][1], marker=marker, color=color, linewidth=1.65, markersize=4)
    a.axhline(0., color='#686868', linewidth=.7)
    a.set(ylabel='Upper ceiling vs. OLS4 (pp)', title='Source-averaged full-grid capacity')
    b.set(ylim=(-1, 35), yticks=[0, 10, 20, 30, 34], ylabel='Recordings with positive ceiling (of 34)',
          title='Where the ceiling permits a gain')
    for ax in (a, b):
        ax.set(xlabel='Maximum edits per query', xticks=x, xticklabels=[*s['caps'][:-1], 'all'])
        ax.axvline(1, color='#777777', linewidth=.6, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    a.legend(loc='upper left', frameon=False, fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == '__main__':
    main()
