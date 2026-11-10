"""Plot the external uncertainty contrast and compile the next anonymous draft."""

import argparse
from io import BytesIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_external_completion_panel as evidence
from scripts import build_m3w_vehicle_pair_angle_candidate as prior


builder = prior.builder
OUT = prior.OUT.parent/'evidence_manuscript_v190_external_bounds'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
FIGURE = 'external_uncertainty.pdf'


def inputs():
    path = evidence.OUT/'readout.json'
    d, h = evidence.storage.load(path)
    if d != evidence.audit():
        raise ValueError('External panel does not replay the frozen receipts')
    return d, dict(external_completion_panel=h, **d['source_hashes'])


def figure(d):
    rows = (
        ('(a) WEPDTOF', d['wepdtof']['curve'], 1., (-.125, .041), [-.12, -.08, -.04, 0., .04],
         'Gain over independent (pp)'),
        ('(b) Songdo', d['songdo']['controls']['independent'], 1000., (-1.56, .34), [-1.5, -1., -.5, 0.],
         r'Gain over independent ($10^{-3}$ pp)'),
    )
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9., 'pdf.fonttype': 42, 'axes.unicode_minus': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 1.47), sharey=True, gridspec_kw={'wspace': .22})
        fig.subplots_adjust(left=.225, right=.985, top=.81, bottom=.34)
        for ax, (title, r, scale, limit, ticks, xlabel) in zip(axes, rows, strict=True):
            lo, hi = r['lower']*scale, r['upper']*scale
            b = [x*scale for x in r['lower_ci95']]
            ax.axvline(0., color='#697074', linestyle=(0, (2., 2.)), linewidth=.8, zorder=0)
            ax.hlines(1., lo, hi, colors='#216873', linewidth=3.2)
            ax.vlines([lo, hi], .90, 1.10, color='#216873', linewidth=1.)
            ax.errorbar(lo, .25, xerr=[[lo-b[0]], [b[1]-lo]], fmt='o', color='#23272a',
                        markersize=3.9, capsize=3.2, elinewidth=1.1, capthick=1.)
            ax.set(xlim=limit, ylim=(-.13, 1.34), xticks=ticks, yticks=[1., .25],
                   yticklabels=['Missing-label bounds', 'Lower-bound 95% CI'])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f' if scale == 1. else '%.1f'))
            ax.set_xlabel(xlabel, labelpad=2)
            ax.set_title(title, loc='left', fontweight='bold', pad=4)
            for key in ('top', 'right', 'left'):
                ax.spines[key].set_visible(False)
            ax.spines['bottom'].set_color('#777777')
            ax.tick_params(axis='x', length=3, pad=2, color='#777777')
            ax.tick_params(axis='y', length=0, pad=4)
        fig.canvas.draw()
        frame = fig.bbox
        for ax in axes:
            for item in [ax.title, ax.xaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                if item.get_visible() and item.get_text():
                    a = item.get_window_extent()
                    if not (frame.x0 <= a.x0 <= a.x1 <= frame.x1 and frame.y0 <= a.y0 <= a.y1 <= frame.y1):
                        raise ValueError('Clipped interval panel label')
        stream = BytesIO()
        fig.savefig(stream, format='pdf', metadata={'CreationDate': None, 'ModDate': None, 'Title': 'External uncertainty'})
        plt.close(fig)
    return stream.getvalue()


def revise(source, d):
    start = 'resolve the sign, and the optimistic endpoint is not a risk pass.'
    songdo = r'[\SongdoLow{}, \SongdoHigh{}]). Joint released'
    if source.count(start) != 1 or source.count(songdo) != 1:
        raise ValueError('External paragraphs changed since the prior verified manuscript')
    r = d['songdo']['controls']['independent']
    graphic = r'''

\begin{figure*}[t]
\centering
\includegraphics[width=.96\textwidth]{external_uncertainty.pdf}
\caption{Joint versus independent selection at matched edit budgets.
The upper bar varies the missing labels within the observed source
sample. The lower bar resamples sources and shows the 95\% bootstrap
interval of the conservative endpoint. WEPDTOF uses 10 of 13 estimable
sources; Songdo uses all 14 source components. The horizontal scales
differ. Positive values favor joint selection.}
\label{fig:external-uncertainty}
\end{figure*}
'''
    explanation = rf'''[\SongdoLow{{}}, \SongdoHigh{{}}]).
For these 14 components the missing-label upper bound is
\({r['upper']:+.6f}\) pp (Fig.~\ref{{fig:external-uncertainty}}).
Missing labels alone cannot change the sign of this sample mean,
but the bootstrap interval of the upper endpoint still includes zero.
Joint released'''
    return source.replace(start, start+graphic, 1).replace(songdo, explanation, 1), (
        'Joint versus independent selection', '0.000102')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    d, hashes = inputs()
    image = figure(d)
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=lambda: (d, hashes),
                  revise_source=revise, extra_assets={FIGURE: image})
    path = OUT/FIGURE
    if path.exists() and path.read_bytes() != image:
        raise FileExistsError('Different verified plot already exists')
    if not path.exists():
        with path.open('xb') as f:
            f.write(image)


if __name__ == '__main__':
    main()
