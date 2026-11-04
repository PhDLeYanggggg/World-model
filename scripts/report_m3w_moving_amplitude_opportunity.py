"""Report every frozen amplitude, never choose a held-score winner."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_opportunity_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_amplitude_opportunity_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row):
    if row['point'] is None or row['CI95'] is None:
        return 'not estimable'
    return f"{row['point']:+.4f} [{row['CI95'][0]:+.4f},{row['CI95'][1]:+.4f}]"


def main():
    a=read(PUBLIC/'analysis.json');reg=read(PUBLIC/'registration.json')
    replay=read(PUBLIC/'verify_execution.json')
    if (not replay['all_scalars_exact'] or a['registration_sha256'] != sha(PUBLIC/'registration.json')
            or not a['all_alpha1_original_policies_and_witnesses_exact']):
        raise ValueError('Complete fixed-grid and exact alpha1/scalar replay required')
    for relative,expected in reg['bindings'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Frozen amplitude definition changed')
    if len(a['feasibility']) != 348 or len(a['policies']) != 138*6:
        raise ValueError('All original views and all six fractions required')
    newly=[r for r in a['feasibility']if r['any_fraction_feasible']and not r['original_binary_feasible']]
    remaining=Counter(r['original_reason']for r in a['feasibility']if not r['any_fraction_feasible'])
    histogram=Counter(r['largest_fraction_with_label_aware_witness']for r in newly)
    fraction_rows=sorted(a['summary'].values(),key=lambda r:r['alpha'])
    evaluate=read(PUBLIC/'evaluate_execution.json')
    lines=['# Fixed Amplitudes Expand Opportunity, Not Deployment','',
        '## Material Passport','',
        'Fresh_run: all six predeclared causal interpolation controls, their',
        'complete risk/utility readouts and separate label-aware candidate-set',
        'witnesses. Cached_verified: original forecasts and data, followed by',
        'a separate exact scalar replay. No neural model or selector is trained.',
        'These are exposed-source development secondary results, not independent',
        'confirmation, validation-selected deployment or a new architecture.','',
        '## Constructive Candidate-Class Result','',
        'The fixed set is alpha={1/32,1/16,1/8,1/4,1/2,1}; every fraction is',
        'reported. Paths use only the existing causal physical floor and saved',
        'past-only neural proposal. No future labels enter path construction.',
        'Labels are used separately for evaluation and the oracle diagnosis.','',
        f"Label-aware finite-risk opportunity increases from 299/348 to **{sum(r['any_fraction_feasible']for r in a['feasibility'])}/348** repeated views.",
        f"The expanded candidate set admits a witness in {len(newly)} of the 34 formerly candidate-limited views.",
        'The largest admissible fraction among these newly feasible views is:', '',
        '| Largest fraction with a label-aware single-row witness | Repeated views |',
        '|---|---:|']
    for alpha,count in sorted(histogram.items()):
        lines.append(f'| {alpha:g} | {count} |')
    lines += ['',f"Still unresolved: {remaining['no_known_positive_easy_reference']} empty-event views and "
        f"{remaining['every_legal_easy_harm_ratio_exceeds_original_budget']} candidate-limited views.",
        'No missing case is removed. A witness chooses a row/fraction with labels;',
        'it is not a causal policy and does not establish learnable coverage.',
        'This candidate opportunity result is not a utility-maximizing oracle.','',
        '## All Uniform Causal Controls','',
        'Use one fixed alpha on the full population, not an oracle-chosen alpha',
        'per agent. Error gain is relative to the same fitting-selected floor.',
        'Intervals use 3,000 source-group resamples after excluded-producer and',
        'actual forecasting-seed averaging, not IID windows. No winning fraction',
        'is chosen or deployed from these scores.','',
        '| Alpha | Known available-ADE gain (%) [95% CI] | Completion-lower gain (%) [95% CI] | Original full-risk support / 348 | Known all/easy violations | Undefined easy |',
        '|---|---:|---:|---:|---:|---:|']
    for r in fraction_rows:
        lines.append(f"| {r['alpha']:g} | {interval(r['known_gain'])} | {interval(r['completion_lower_gain'])} | "
            f"{r['supported_record_views']}/348 | {r['known_all_risk_violations']}/{r['known_easy_risk_violations']} | {r['undefined_easy_views']} |")
    lines += ['','Every uniform control fails the full original contract. Even alpha=1/32',
        'retains 65 known easy-risk violations and passes only 96/348 repeated',
        'views. Smaller path changes are not a safety certificate. A nonzero',
        'path change still counts as an intervention; it is not relabeled as',
        'fallback just because the amplitude is small. Positive source utility',
        'intervals do not override risk failure or establish independent success.','',
        '## What Is And Is Not Repaired','',
        'The candidate-class limitation is partly constructive: intermediate',
        'paths allow many witnesses that the full binary proposal lacked.',
        'But fixed interpolation is not a learned safe admission method, and',
        'the empty-event support problem persists. The next method test would',
        'need past-conditioned amplitude cost/risk learning and controlled',
        'joint allocation, with independent model selection/calibration and',
        'final confirmation. It cannot inherit old full-amplitude cost scores',
        'unchanged for new paths or use diagnostic witness choices as inputs.','',
        'Do not extend this grid from exposed outcomes, pick the most favorable',
        'fraction/seed or silently change the population, event or 2% budget.',
        'Any later training must prospectively freeze its action class and',
        'recompute candidate-dependent causal features and supervised moments.',
        'The 20 unused source groups remain unadmitted pending the requested',
        'scientific role decision; DUT and DroneCrowd roles remain unchanged.','',
        '## Reproduction And Limits','',
        f"Evaluation PID {evaluate['pid']}: {evaluate['seconds']:.6f} s, observed exit 0.",
        f"Separate replay PID {replay['pid']}: {replay['seconds']:.6f} s, observed exit 0.",
        'Every alpha=1 per-source policy and binary witness matches its parent',
        'exactly. All new scalars replay. The original 195,286 rows, 23 supported',
        'groups and 348 repeated recording views remain. Original 24-group primary',
        'is still not estimable. Partial available-label ADE and bounds on wholly',
        'unknown outcomes do not identify the missing steps of a full 12-step path.',
        'Pixels/native annotation steps are not verified metres or seconds.',
        'No new model, deployment, true3D, foundation, physical-safety or',
        'submission-ready claim. Stage5C execution and SMC stay off.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),sharey=True)
    for ax,key,title in zip(axes,('known_gain','completion_lower_gain'),('Known available-label gain','Missing-outcome completion lower')):
        for i,r in enumerate(fraction_rows):
            v=r[key]
            ax.errorbar(v['point'],i,xerr=[[v['point']-v['CI95'][0]],[v['CI95'][1]-v['point']]],fmt='o',capsize=3,color='#176e8b')
        ax.axvline(0,color='0.5',linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel('Gain over the same floor (%), grouped 95% CI')
    axes[0].set_yticks(range(6),[f"alpha={r['alpha']:g}"for r in fraction_rows])
    axes[0].invert_yaxis()
    fig.suptitle('Six fixed causal controls; none passes the complete risk contract')
    fig.tight_layout()
    (PUBLIC/'figures').mkdir(exist_ok=True)
    (PRIVATE/'figure_previews').mkdir(exist_ok=True)
    fig.savefig(PUBLIC/'figures/fixed_amplitudes.svg')
    fig.savefig(PRIVATE/'figure_previews/fixed_amplitudes.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
