"""Render the exact training-replay diagnosis, without outcome re-scoring."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_admission_monitor as run


def build():
    complete = json.loads((run.PUBLIC / 'complete.json').read_text())
    summary = json.loads((run.PUBLIC / 'summary.json').read_text())
    if complete['summary_sha256'] != run.parent.sha(run.PUBLIC / 'summary.json'):
        raise ValueError('Frozen telemetry summary changed')
    rows = []
    for ref in complete['replays']:
        path = ROOT / ref['path']
        if run.parent.sha(path) != ref['sha256']:
            raise ValueError('Frozen replay receipt changed')
        row = json.loads(path.read_text())
        if run.parent.sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Private diagnostic checkpoint changed')
        rows.append(row)
    contexts = sorted(json.loads(run.REG.read_text())['contexts'])
    expected = {(c, seed, 'admitted_score') for c in contexts for seed in (17, 29, 43)}
    expected.add((contexts[0], 17, 'upper_score'))
    keys = [(r['context'], r['seed'], r['objective_arm']) for r in rows]
    if len(keys) != 19 or len(set(keys)) != 19 or set(keys) != expected:
        raise ValueError('The exact19endpoint roster is required')
    if run.summarize(rows) != summary:
        raise ValueError('Complete19replay summary reconstruction differs')
    return complete, summary, rows


def main():
    complete, s, rows = build()
    t = s['admitted_training_totals']
    ratio = s['snapshot_gradient_distributions']
    lines = ['# Why did the unit admission penalty barely change the policy?', '',
        '## Material Passport', '',
        'Fresh run: exact instrumented Torch training replay on cached-verified SOURCE fitting inputs.',
        'No new candidate, policy score, held/calibration outcome or independent test.', '',
        f"All 19 complete 2,000-step replays match the frozen scientific states. PID {complete['pid']}",
        f"takes {complete['seconds']:.3f}s at {complete['peak_RSS_bytes']}B peak RSS. The first full",
        'upper control also serves as the measured local resource pilot. All original weights,',
        'optimizer states, RNG, draws and original loss traces stay exact; added telemetry is separate.', '',
        '## Activation is frequent, but gradient pressure is tiny', '',
        f"The 18 admitted trajectories contain {t['step']} updates, {t['row_draws']} repeated row draws,",
        f"{t['unknown_draws']} unknown draws and zero observed-cost supervision on unknown outcomes.",
        f"The auxiliary is positive on {t['steps_aux_positive']}/{t['step']} steps ({100*t['steps_aux_positive']/t['step']:.4f}%).",
        f"Its first activation is between steps {min(s['first_aux_positive_steps'])} and {max(s['first_aux_positive_steps'])}.",
        'A missing/late-activation explanation is therefore not supported by these actual trajectories.', '',
        f"Summed auxiliary / summed anchor loss is {100*s['auxiliary_over_anchor_loss']:.6f}%.",
        f"Across {s['snapshots']} fixed step1/every50/final snapshots, the median auxiliary/anchor",
        f"parameter-gradient norm ratio is {100*ratio['auxiliary_over_anchor_gradient']['quantiles'][2]:.6f}%",
        f"(p90 {100*ratio['auxiliary_over_anchor_gradient']['quantiles'][3]:.6f}%,",
        f"maximum {100*ratio['auxiliary_over_anchor_gradient']['quantiles'][4]:.6f}%).",
        f"The easy-channel median ratio is {100*ratio['easy_auxiliary_over_anchor_gradient']['quantiles'][2]:.6f}%.",
        f"The total auxiliary/anchor cosine has median {ratio['auxiliary_anchor_cosine']['quantiles'][2]:.8f};",
        'some gradients conflict, but conflict alone is not the measured magnitude bottleneck.', '',
        f"Among {s['snapshot_false_easy_gate_defined']} defined false-easy selected snapshot rows,",
        f"{s['snapshot_false_easy_gate_under_001']} assign below1%of total predicted harm to the easy branch.",
        'This is consistent with a nested-output bottleneck, but does not uniquely distinguish',
        'loss scaling, output parameterization, conditional feature quality or label noise.', '',
        '## Controlled next action', '',
        'Test one fixed gradient-norm balancing rule on the same model and losses. Give the',
        'existing admission auxiliary the same pre-clipping parameter-gradient norm as the',
        'unchanged anchor when nonzero, with a registered numerical cap. Do not change the',
        'decoder, inputs, 2% risk rule, sampling or reserved roles. Reuse exact frozen controls',
        'rather than claiming they were newly trained. This is an optimization repair, not',
        'a new deferral framework or evidence that safety will pass. No threshold sweep.', '',
        'The old failed utility/safety conclusions remain unchanged. These are correlated fitting',
        'draws and training snapshots, not independent samples or population uncertainty.',
        'No deployment, submission-ready, metric/seconds, true3D or foundation claim.',
        'No Stage5C execution or SMC.', '']
    (run.PUBLIC / 'readout.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                        'svg.hashsalt': run.NAME, 'svg.fonttype': 'none'})
    admitted = [r['telemetry'] for r in rows if r['objective_arm'] == 'admitted_score']
    steps = [v['step'] for v in admitted[0]['rows']]
    if any([v['step'] for v in m['rows']] != steps for m in admitted):
        raise ValueError('All snapshots must follow the same fixed schedule')
    gradients = np.array([[v['auxiliary_over_anchor_gradient'] for v in m['rows']] for m in admitted])
    easy = np.array([[v['easy_auxiliary_over_anchor_gradient'] for v in m['rows']] for m in admitted])
    anchor = np.array([[v['losses']['base']+v['losses']['upper'] for v in m['rows']] for m in admitted])
    auxiliary = np.array([[v['losses']['aux'] for v in m['rows']] for m in admitted])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    low, med, high = np.quantile(gradients, [.1, .5, .9], axis=0)*100
    ax[0].fill_between(steps, low, high, color='#007C91', alpha=.15)
    ax[0].plot(steps, med, color='#007C91', label='Total auxiliary')
    ax[0].plot(steps, np.median(easy, axis=0)*100, color='#8B3F72', label='Easy channel')
    ax[0].set_ylabel('Auxiliary / anchor gradient norm (%)')
    ax[0].set_title('Parameter-gradient pressure: median and 10-90%', fontsize=11)
    for matrix, name, color in ((anchor, 'Anchor', '#555555'), (auxiliary, 'Admission auxiliary', '#B44D48')):
        value = np.median(matrix, axis=0)
        ax[1].plot(steps, np.where(value > 0, value, np.nan), label=name, color=color)
    ax[1].set_yscale('log'); ax[1].set_ylabel('Median training loss')
    ax[1].set_title('Actual losses; zero auxiliary at step1 omitted', fontsize=11)
    for panel in ax:
        panel.set_xlabel('Original optimizer step'); panel.legend(frameon=False); panel.grid(alpha=.15)
    fig.suptitle('Exact replay of 18 admitted heads: exposed fitting data only', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    folder = run.PUBLIC / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'optimization_pressure.svg', metadata={'Date': None}, bbox_inches='tight')
    fig.savefig(run.PRIVATE / 'optimization_pressure.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    run.parent.write(run.PUBLIC / 'evidence.json', dict(summary_sha256=run.parent.sha(run.PUBLIC / 'summary.json'),
        complete_sha256=run.parent.sha(run.PUBLIC / 'complete.json'),
        report_code_sha256=run.parent.sha(Path(__file__)),
        readout_sha256=run.parent.sha(run.PUBLIC / 'readout.md'),
        figure_sha256=run.parent.sha(folder / 'optimization_pressure.svg'),
        all19_scientific_states_exact=True, new_policy_evidence=False,
        effectiveness_recomputed=False, independent_confirmation=False, deployment_changed=False))
    print(json.dumps(dict(replays=19, exact=True, new_policy=False)))


if __name__ == '__main__':
    main()
