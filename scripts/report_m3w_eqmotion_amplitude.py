"""Rebuild the fixed amplitude readout from verified public scalars only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_eqmotion_amplitude_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_eqmotion_amplitude_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    read = lambda p: json.loads(p.read_text())
    s = read(PUBLIC/'summary.json'); v = read(PUBLIC/'verification.json'); complete = read(PUBLIC/'complete.json')
    if (not v['metric_replay_exact'] or v['groups'] != 72 or v['policy_group_evaluations'] != 1440
            or v['summary_sha256'] != sha(PUBLIC/'summary.json')):
        raise ValueError('The complete 1,440-group metric replay must finish')
    for path, h in complete['groups_manifest'].items():
        if sha(ROOT/path) != h: raise ValueError('Group scalar receipt changed')
    fmt = lambda r: 'not estimable' if r['point'] is None else f'{r["point"]:+.4f} [{r["CI95"][0]:+.4f}, {r["CI95"][1]:+.4f}]'
    lines = ['# Fixed-Support Amplitude Readout', '',
        '## Material Passport', '',
        'Fresh_run: complete registered SOURCE amplitude reductions and a separate',
        'exact replay. Cached_verified: the nine EqMotion/current-PCT forecast pairs,',
        'original 149,510-window whitelist, roles and fitting-site cutoffs. This',
        'report is public-scalar reconstruction, not new fitting or Torch inference.', '',
        '## Complete Curve', '',
        '|Alpha|EqMotion known gain vs floor % [95% CI]|EqMotion completion-lower % [95% CI]|EqMotion supported/444|PCT known gain %|PCT supported/444|',
        '|---:|---:|---:|---:|---:|---:|']
    for alpha in s['amplitudes']:
        r = s['curve'][str(alpha)]; a = r['arms']['eqmotion']; b = r['arms']['history_PCT']
        lines.append(f'|{alpha}|{fmt(a["known_gain"])}|{fmt(a["completion_lower_gain"])}|{a["safety"]["supported_record_views"]}/444|{b["known_gain"]["point"]:+.4f}|{b["safety"]["supported_record_views"]}/444|')
    q = s['curve']['0.25']; a = q['arms']['eqmotion']; risk = a['safety']
    lines += ['', 'Gains use the same producer-fitting-selected causal floor, not the CV',
        'denominator from the preceding forecasting paper. They must not be mixed.',
        'Intervals use 3,000 locality resamples after two excluded producer folds and',
        'actual seeds17/29/43 are averaged. All twelve localities and 74 recordings',
        'remain. The 444 recording/pipeline views are repeated views, not 444 scenes.', '',
        '## Predeclared Quarter-Amplitude Check', '',
        f'Known gain: {fmt(a["known_gain"])}%. Completion lower: {fmt(a["completion_lower_gain"])}%.',
        f'EqMotion minus matched-amplitude current PCT: {fmt(q["EqMotion_minus_current_PCT"]["known_pp"])} pp.',
        f'The corresponding direct paired lower is {fmt(q["EqMotion_minus_current_PCT"]["completion_lower_pp"])} pp.',
        f'Full recording-risk support: {risk["supported_record_views"]}/444. Known all/easy',
        f'violations: {risk["known_all_violations"]}/{risk["known_easy_violations"]}; completion all/easy',
        f'violations: {risk["completion_all_violations"]}/{risk["completion_easy_violations"]}.',
        f'Worst signed easy-net degradation is {risk["worst_signed_easy_net_percent"]:.4f}%.', '',
        'The mean easy gain is positive, but the worst recording is not preserved.',
        'Small average error and nonempty support do not imply the complete 2% pass.', '',
        '## Failure Mechanism', '',
        'Every positive grid amplitude has defined all/easy/easy-net references in',
        'every required view: fixed support removes the empty-denominator failure.',
        'No positive grid point passes the complete contract. At alpha .001, known',
        'all/easy and signed easy-net satisfy 2%, but six completion-easy checks fail.',
        'All six are the same recording51 at locality082, across two excluded folds',
        'and three actual seeds, not six independent failure scenes. The recording',
        'has 78 target windows, 16 wholly unknown futures and little known easy',
        'reference mass. The bound, not an observed error on an unknown future, fails.',
        'Quarter amplitude additionally has substantial observed easy-harm violations.',
        'Thus unknown outcomes are not the sole problem at useful amplitudes.', '',
        'Alpha .001 yields only 0.0274% known gain. Its positive mean lower interval',
        'does not rescue the six original recording failures. The grid has not proved',
        'that every arbitrarily smaller amplitude fails, or that no causal policy can',
        'work. Zero amplitude is abstention, not a supported learned intervention.',
        'Every positive amplitude has 100% active rows here; amplitude magnitude, not',
        'binary switch count, explains the near-fallback regime.', '',
        '## Consequence For The Next Experiment', '',
        'Do not tune another global threshold or simply add model width. First test',
        'whether past motion and detector-history fields can predict released future',
        'label support on SOURCE-excluded localities. If that signal exists, compare',
        'ordinary known-only costs with explicit unknown-disagreement upper-bound',
        'targets under the same original risk contract. These are bounds/observation',
        'process targets, not invented future trajectory labels or physical visibility.',
        'The current study does not train or validate that repair. No future-valid',
        'mask may enter inference, and recording51/locality082 must not be excluded.', '',
        '## Compute And Limits', '',
        f'Primary reduction PID{complete["pid"]}: {complete["seconds"]:.3f} s, peak {complete["peak_RSS_bytes"]:,} bytes.',
        f'Separate metric replay PID{v["pid"]}: {v["seconds"]:.3f} s, peak {v["peak_RSS_bytes"]:,} bytes.',
        'All72 producer/locality groups and1,440 family/amplitude policy evaluations',
        'reproduce exactly. New optimizer updates:0. No new Torch inference is claimed.',
        'Partial-future available-label ADE is not full12 ADE; completion bounds cover',
        'wholly unknown rows only. SOURCE is exposed development, not independent',
        'final confirmation. No calibration/private final/DroneCrowd arrays are read.',
        'No selected deployment alpha, safety promotion or submission readiness.',
        'Pixels/rawstride12/released silver, not metric, seconds, human gold, physical',
        'safety, true3D or foundation. Stage5C and SMC remain off.', '']
    (PUBLIC/'readout.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                         'svg.hashsalt':'m3w_eqmotion_amplitude', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    x = s['amplitudes'][1:]
    for family, label, color in (('eqmotion', 'EqMotion', '#187f8a'), ('history_PCT', 'Current PCT', '#b17e31')):
        arms = [s['curve'][str(a)]['arms'][family] for a in x]
        y = [r['known_gain']['point'] for r in arms]
        axes[0].plot(x, y, 'o-', label=label, color=color, markersize=4)
        axes[0].fill_between(x, [r['known_gain']['CI95'][0] for r in arms],
                             [r['known_gain']['CI95'][1] for r in arms], alpha=.15, color=color)
        axes[1].plot(x, [r['safety']['supported_record_views'] for r in arms], 'o-', label=label, color=color, markersize=4)
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Fixed intervention amplitude')
        ax.grid(alpha=.2); ax.axvline(.25, color='#888888', linestyle=':', linewidth=1)
    axes[0].set(ylabel='Known ADE gain over fixed floor (%)', title='Observed utility: locality-bootstrap 95% CI')
    axes[1].axhline(444, color='#777777', linestyle='--', linewidth=1, label='All 444 required views')
    axes[1].set(ylabel='Views satisfying full original contract', title='No positive grid point supports every view', ylim=(-10, 465))
    axes[0].legend(loc='upper left'); axes[1].legend(loc='lower left')
    fig.suptitle('Attenuation restores reference support, not useful full-risk compliance', fontsize=11)
    directory = PUBLIC/'figures'; directory.mkdir(exist_ok=True)
    fig.savefig(directory/'amplitude_curve.svg', metadata={'Date':None})
    fig.savefig(PRIVATE/'amplitude_curve.png', dpi=150); plt.close(fig)
    receipt = dict(result_source='verified_public_scalars_fresh_report_and_render_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'), verification_sha256=sha(PUBLIC/'verification.json'),
        reporter_sha256=sha(Path(__file__)), svg_sha256=sha(directory/'amplitude_curve.svg'),
        private_preview_sha256=sha(PRIVATE/'amplitude_curve.png'), preview_visually_checked=False,
        deployment_changed=False, submission_ready=False)
    (directory/'sources.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
