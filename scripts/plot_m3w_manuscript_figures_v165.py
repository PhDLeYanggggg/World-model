"""Vector small multiples and interval plot from the frozen v164 aggregates."""
from matplotlib import rc_context
from matplotlib.figure import Figure

from scripts.plot_m3w_manuscript_figures import INK, STYLE, error, save
from scripts.plot_m3w_decision_bottlenecks import load

PRINT = {**STYLE, 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
         'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '.40',
         'xtick.color': INK, 'ytick.color': INK}


def rows(ax, labels):
    ax.set_ylim(len(labels) - .53, -.47)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis='y', pad=5)


def count_marks(ax, counts, *, end, ticks):
    ax.set_xlim(0, end)
    ax.set_xticks(ticks)
    for y, value in enumerate(counts):
        ax.hlines(y, 0, value, color='.66', lw=.75)
        ax.plot(value, y, 'o', ms=3.4, color=INK, clip_on=False)
        ax.annotate(f'{value:,}', (value, y), xytext=(4, 0),
                    textcoords='offset points', va='center')


def decision(path):
    retrieval = load('native_recording_witness_retrieval_v1', 'retention.json', 'retention_execution.json')
    portfolio = load('native_joint_path_portfolio_v1', 'analysis.json', 'replay_execution.json')
    capacity = [retrieval['stages'][k] for k in ('retrieved_bank', 'decoded_queries', 'admitted_queries')]
    policies = [portfolio['grouped']['risk'][k] for k in ('single_gated', 'matched_joint', 'screen_first_joint')]
    if any(c['unresolved'] for c in capacity) or any(p['required_views'] != 102 for p in policies):
        raise ValueError('Require the complete development roster')

    with rc_context(PRINT):
        fig = Figure(figsize=(7.20, 1.76))
        options = ('Retrieved bank', 'One decoded option', 'Admitted options')
        names = [f'{name} ({p["switches"]:,})' for name, p in
                 zip(('Original', 'Matched joint', 'Screen first'), policies, strict=True)]
        for x, title in ((.16, '(a) Hindsight capacity'), (.65, '(b) Past-only policies')):
            fig.text(x, .89, title, fontsize=10, weight='semibold')
        for x, title, field, data, labels in (
            (.16, 'Risk feasible', 'feasible', capacity, options),
            (.33, 'Positive gain bound', 'positive', capacity, ('', '', '')),
            (.65, 'All-risk > 2%', 'all_risk_over_2_views', policies, names),
            (.82, 'All checks pass', 'supported_views', policies, ('', '', '')),
        ):
            ax = fig.add_axes((x, .30, .143, .41))
            rows(ax, labels)
            ax.set_title(title, loc='left', pad=5)
            count_marks(ax, [d[field] for d in data], end=102, ticks=(0, 50, 100))
        fig.text(.565, .055, 'Recording-seed views (out of 102)', ha='center')
        save(fig, path)


def coverage(path, readout, matched):
    donors = readout['donors']
    count = [donors[k]['same_action_overlap'] for k in ('old', 'direct', 'conditional')]
    total = donors['old']['actual']
    if total != 2453 or not all(0 <= n <= total for n in count):
        raise ValueError('Require the coverage roster stated in the caption')

    with rc_context(PRINT):
        fig = Figure(figsize=(3.4, 3.10))
        a, b = fig.subplots(2, 1)
        fig.subplots_adjust(left=.232, right=.952, top=.89, bottom=.22, hspace=.87)
        rows(a, ('Frozen', 'Direct', 'Conditional'))
        a.set_title('(a) Seed actions recovered', loc='left', pad=6)
        a.axvline(total, color='.43', lw=.65, ls=(0, (3, 3)))
        count_marks(a, count, end=2600, ticks=(0, 1000, 2000))
        a.set_xlabel('Query views', labelpad=2)

        rows(b, ('Direct', 'Conditional'))
        b.set_title('(b) Matched-count gain', loc='left', pad=6)
        b.axvline(0, color='.43', lw=.65, ls=(0, (3, 3)))
        for label, offset, marker, field in (('Known steps', -.15, 'o', 'known_pp'),
                                              ('Full-grid lower', .15, 's', 'completion_lower_pp')):
            for y, method in enumerate(('direct', 'conditional')):
                z = matched['summary']['effects'][method][field]
                error(b, z['point'], y + offset, z['interval95'], marker,
                      filled=marker == 'o', label=label if y == 0 else None)
        b.set_xlabel('Gain over frozen rule (pp)', labelpad=2)
        fig.legend(*b.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.232, .035),
                   ncol=2, frameon=False, fontsize=8.5, handlelength=1.35, handletextpad=.45,
                   columnspacing=1.3, borderaxespad=0)
        save(fig, path)
