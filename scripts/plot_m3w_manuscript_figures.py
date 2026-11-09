"""Print-friendly figures from the manuscript's unchanged aggregate data."""
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from scripts import build_m3w_decision_manuscript_v66 as opendd
from scripts.plot_m3w_decision_bottlenecks import load

STYLE = {
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'font.size': 8.5, 'pdf.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.spines.left': False,
    'axes.linewidth': .65, 'xtick.major.width': .65, 'xtick.major.size': 3,
    'ytick.major.size': 0, 'hatch.linewidth': .55, 'figure.facecolor': 'white',
}
INK = '0.15'
GRAY = '0.42'
META = {'Creator': '', 'Producer': '', 'CreationDate': None, 'ModDate': None}


def save(fig, path):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    bbox = fig.get_tightbbox(canvas.get_renderer())
    w, h = fig.get_size_inches()
    if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > w or bbox.y1 > h:
        raise ValueError(f'Figure text extends outside the page: {path.name}, {bbox.bounds}')
    fig.savefig(path, metadata=META)


def panel(ax, title, labels):
    ax.set_title(title, loc='left', fontsize=9, pad=6)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.5, -.5)
    ax.tick_params(axis='y', pad=5)
    ax.set_axisbelow(True)


def counts(ax, n, label='Recording-seed views'):
    ax.set_xlim(0, n)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlabel(label, labelpad=4)


def comparison(ax, data, fields, labels, markers):
    for j, (field, label, mark) in enumerate(zip(fields, labels, markers, strict=True)):
        y = np.arange(len(data)) + (-.15 if j == 0 else .15)
        x = [v[field] for v in data]
        ax.plot(x, y, linestyle='none', marker=mark, ms=4.7, color=INK,
                markerfacecolor='white' if j == 0 else INK, label=label, clip_on=False)
        for xx, yy in zip(x, y, strict=True):
            ax.annotate(str(xx), (xx, yy), xytext=(5, 0), textcoords='offset points', va='center')
    ax.legend(loc='upper left', bbox_to_anchor=(-.015, -.48), frameon=False,
              fontsize=8, handletextpad=.35, borderaxespad=0)


def decision(path):
    retrieval = load('native_recording_witness_retrieval_v1', 'retention.json', 'retention_execution.json')
    portfolio = load('native_joint_path_portfolio_v1', 'analysis.json', 'replay_execution.json')
    cap = [retrieval['stages'][k] for k in ('retrieved_bank', 'decoded_queries', 'admitted_queries')]
    risk = [portfolio['grouped']['risk'][k] for k in ('single_gated', 'matched_joint', 'screen_first_joint')]
    if any(r['required_views'] != 102 for r in risk) or any(c['unresolved'] for c in cap):
        raise ValueError('Require all fixed development views')
    with rc_context(STYLE):
        fig = Figure(figsize=(7.3, 2.37))
        a, b = fig.subplots(1, 2)
        fig.subplots_adjust(left=.20, right=.97, top=.88, bottom=.42, wspace=.78)
        panel(a, '(a) Hindsight capacity', ('Retrieved bank', 'One decoded option', 'Admitted options'))
        panel(b, '(b) Past-only policies',
              [f'{k} ({v["switches"]:,})' for k, v in zip(('Original', 'Matched joint', 'Screen first'), risk, strict=True)])
        comparison(a, cap, ('feasible', 'positive'), ('Risk feasible', 'Positive lower gain vs. OLS4'), ('s', 'o'))
        comparison(b, risk, ('all_risk_over_2_views', 'supported_views'),
                   ('All-risk violation', 'All checks pass'), ('s', 'o'))
        for ax in (a, b):
            counts(ax, 112)
            ax.set_xticks((0, 25, 50, 75, 102))
        save(fig, path)


def recipient(path):
    fixed = load('native_fixed_recipient_risk_bound_v1', 'analysis.json', 'replay_execution.json')
    exact = load('native_exact_count_joint_capacity_v1', 'analysis.json', 'replay_execution.json')
    policy = load('native_recipient_shortfall_v1', 'analysis.json', 'replay_execution.json')
    n = exact['views']
    values = [(0, v['not_ruled_out'], v['counts']['provably_infeasible'])
              for k in ('fixed_neural_count', 'free_neural_count') for v in [fixed['bounds'][k]]]
    values += [(v['verified_feasible'], v['unresolved'], v['proved_infeasible'])
               for k in ('fixed_nomination', 'full_forecast_bank') for v in [exact['risk_capacity'][k]]]
    d = [policy['grouped']['risk'][k] for k in ('gain', 'support_first', 'shortfall')]
    if n != 102 or fixed['views'] != n or policy['grouped']['views'] != n or any(sum(row) != n for row in values):
        raise ValueError('Require the complete common recipient roster')
    if len({v['switches'] for v in d}) != 1:
        raise ValueError('Require matched counts')
    with rc_context(STYLE):
        fig = Figure(figsize=(7.3, 2.37))
        a, b = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.23, 1.]})
        fig.subplots_adjust(left=.29, right=.97, top=.88, bottom=.38, wspace=.68)
        panel(a, '(a) Hindsight feasibility', ('Fixed people, fixed neural count', 'Fixed people, free neural count',
                                                  'All people, one path each', 'All people, all paths'))
        panel(b, '(b) Past-only policy passes', ('Gain', 'Support-first', 'Risk shortfall'))
        edges = np.zeros(len(values))
        for width, name, fill, hatch in zip(np.asarray(values).T, ('Verified feasible', 'Not ruled out', 'Proved infeasible'),
                                             ('0.25', 'white', '0.79'), ('', '////', ''), strict=True):
            a.barh(range(len(values)), width, left=edges, height=.51, facecolor=fill,
                   edgecolor=INK, linewidth=.55, hatch=hatch, label=name)
            for i, (left, value) in enumerate(zip(edges, width, strict=True)):
                if value:
                    a.text(left+value/2, i, str(value), ha='center', va='center',
                           color='white' if fill == '0.25' else INK,
                           bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .35} if hatch else None)
            edges += width
        a.legend(loc='upper left', bbox_to_anchor=(-.005, -.39), ncol=2, frameon=False,
                 fontsize=7.6, handlelength=1.45, handletextpad=.4, columnspacing=.85, borderaxespad=0)
        for j, row in enumerate(d):
            x = row['supported_views']
            b.plot([0, x], [j, j], color='.73', lw=.7)
            b.plot(x, j, 'o', ms=4.5, color=INK, clip_on=False)
            b.annotate(f'{x}/{n}', (x, j), xytext=(5, 0), textcoords='offset points', va='center')
        for ax in (a, b):
            counts(ax, n)
            ax.set_xticks((0, 25, 50, 75, 102))
        save(fig, path)


def error(ax, value, row, interval, marker='o', filled=True, label=None):
    lo, hi = interval
    ax.errorbar(value, row, xerr=[[value-lo], [hi-value]], fmt=marker, ms=4, lw=.85,
                capsize=2.2, color=INK, markerfacecolor=INK if filled else 'white', label=label)


def capacity(path):
    _, u, cap, count = opendd.evidence()
    values = [cap[k]['verified_positive_both'] for k in ('full_total', 'full_stream', 'nested_total', 'nested_stream')]
    values.append(count['dual_positive'])
    with rc_context(STYLE):
        fig = Figure(figsize=(7.3, 2.1))
        a, b = fig.subplots(1, 2)
        fig.subplots_adjust(left=.15, right=.97, top=.86, bottom=.25, wspace=.75)
        panel(a, '(a) Capacity and realized passes', ('Full / total', 'Full / online', 'Nested / total',
                                                       'Nested / online', 'Learned count'))
        for j, x in enumerate(values):
            a.plot([0, x], [j, j], color='.72', lw=.65)
            a.plot(x, j, marker='s' if j == 4 else 'o', ms=4.5, color=INK)
            a.annotate(str(x), (x, j), xytext=(5, 0), textcoords='offset points', va='center')
        counts(a, 606, 'Risk pass and beat both baselines (/606)')
        panel(b, '(b) Added value of peer context', ('vs. motion (online)', 'vs. motion (matched)', 'vs. prefix proxy'))
        b.axvline(0, color=GRAY, linestyle=(0, (3, 3)), lw=.7)
        for j, k in enumerate(('peers_vs_motion', 'peers_vs_motion_matched', 'peers_vs_proxy')):
            z = u['direct'][k]
            error(b, z['point'], j, z['interval95'])
        b.set_xlim(-.10, .20)
        b.set_xlabel('Paired gain (pp; 95% location interval)', labelpad=4)
        save(fig, path)


def signed_coverage(path, r, m):
    donor = r['donors']
    n = [donor[k]['same_action_overlap'] for k in ('old', 'direct', 'conditional')]
    with rc_context(STYLE):
        fig = Figure(figsize=(3.4, 3.55))
        a, b = fig.subplots(2, 1)
        fig.subplots_adjust(left=.26, right=.96, top=.91, bottom=.22, hspace=.92)
        panel(a, '(a) Correctly recovered seeds', ('Frozen', 'Direct', 'Conditional'))
        a.axvline(donor['old']['actual'], color=GRAY, lw=.7, linestyle=(0, (3, 3)))
        for j, x in enumerate(n):
            a.plot([0, x], [j, j], color='.72', lw=.65)
            a.plot(x, j, 'o', color=INK, ms=4.5)
            a.annotate(f'{x:,}', (x, j), xytext=(5, 0), textcoords='offset points', va='center')
        counts(a, 2700, 'True seed queries')
        panel(b, '(b) Matched-count gain', ('Direct', 'Conditional'))
        b.axvline(0, color=GRAY, lw=.7, linestyle=(0, (3, 3)))
        for tag, off, mark, field in (('Known labels', -.13, 'o', 'known_pp'),
                                      ('Full-grid lower', .13, 's', 'completion_lower_pp')):
            for j, k in enumerate(('direct', 'conditional')):
                z = m['summary']['effects'][k][field]
                error(b, z['point'], j+off, z['interval95'], mark, mark=='o', tag if j==0 else None)
        b.set_xlabel('Swap minus frozen gain (pp)', labelpad=4)
        b.legend(loc='upper left', bbox_to_anchor=(-.32, -.55), ncol=2,
                 frameon=False, fontsize=8, handletextpad=.45, columnspacing=1.)
        save(fig, path)
