"""Render separately verified fixed-selection score accounting,not policy success."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import probe_m3w_selected_scores as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    p = run.PUBLIC; c, s, v = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not v['all72groups_exact'] or v['summary_sha256'] != run.sha(p/'summary.json')
            or v['complete_sha256'] != run.sha(p/'complete.json')
            or c['execution_inputs_sha256'] != run.sha(p/'execution_inputs.json')):
        raise ValueError('Separate complete72-group score replay required')
    for rel, digest in json.loads(run.REG.read_text())['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Frozen score definition changed')
    groups = aggregate_groups(run.NAME); rows = [groups[Path(ref['path']).name] for ref in c['groups']]
    if run.summarize(rows) != s or not s['all288_original_selected_MSE_controls_close']:
        raise ValueError('Canonical72-group reduction or original MSE controls differ')
    verify(ROOT, p)
    return dict(summary=s, complete=c, verification_runtime=json.loads((p/'verification_runtime.json').read_text()))


def render(value):
    s = value['summary']; lines = ['# Original versus final selected-score errors', '', '## Material Passport', '',
        'fresh_run:all72 groups of critic inference and label-aware arithmetic,plus a separate exact complete diagnostic replay.',
        'cached_verified:four frozen critic families,source packets and original/final policy choices. No choice is reoptimized.',
        f"Run PID{value['complete']['pid']}: {value['complete']['seconds']:.3f}s;diagnostic replay PID{value['verification_runtime']['pid']}: {value['verification_runtime']['seconds']:.3f}s.",
        'All288 original-set MSE controls match existing readouts within registered floating tolerance.',
        'This replays diagnosis,not the optimizer;it is not a replacement for either complete policy replay.',
        'No model,threshold,decision rule,risk criterion or data role changes.', '',
        '## Query support and changed selections', '',
        '|Critic / selected set|Selected row views|Complete queries|Unknown queries|Empty queries|Introduced rows|Changed actions|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in s['variants'].items():
        c = r['counts']; lines.append(f"|{name}|{c['selected_rows']}|{c['complete_selected_queries']}|{c['unknown_selected_queries']}|"
            f"{c['empty_selected_queries']}|{c['introduced_rows']}|{c['changed_action_rows']}|")
    lines += ['', 'Complete means every selected row has five available ADE-derived cost labels,not that every future point or endpoint exists.',
        'Counts are reused context/head-seed views,not independent agents or scenes.',
        'All causal rows remain. Unknown/empty means are not imputed as zero.', '',
        '## Conditional score accounting', '',
        'Scores use the same causal-scale units,not different training-loss denominators.',
        'Average supported context/seed views within locality,then supported localities.',
        'Support differs by final selection;these conditional means are NOT full-cohort or causal effects.', '',
        '|Critic / set|Channel|Score MSE|Mean underprediction|Harm component|Reference component|Supported views/localities|Pred<=0,true>0 queries|Zero true references|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in s['variants'].items():
        moments = r['moments']; support = moments['mean_score_squared_error']
        for channel, label in enumerate(('all', 'easy')):
            def fmt(field):
                v = moments[field]['mean']; return f'{v[channel]:.9g}' if v is not None else 'not estimable'
            lines.append(f"|{name}|{label}|{fmt('mean_score_squared_error')}|{fmt('mean_signed_score_underprediction')}|"
                f"{fmt('mean_harm_underprediction')}|{fmt('mean_reference_bias_contribution')}|"
                f"{support['supported_views']}/72;{support['supported_localities']}/12|"
                f"{r['predicted_signed_safe_observed_positive_excess'][channel]}|{r['zero_true_reference_complete_queries'][channel]}|")
    lines += ['', 'The identity is: true-minus-pred signed score = true-minus-pred harm plus',
        '0.02 times pred-minus-true reference,each divided by summed causal scale.',
        'Positive reference contribution means reference inflation also understates the signed risk.',
        'This is arithmetic,not a unique causal attribution or conditional-mean misspecification proof.',
        'Earlier candidate-invariant reference sharing also failed its matched study;reference bias here',
        'does not establish incoherence as a sufficient explanation or a new successful repair.',
        'Strict sign counts are not the ratio-support/finite-record2% gate;zero reference remains undefined.',
        'All original risk requirements remain in the main complete policy readouts.', '',
        '## Interpretation boundary', '',
        'Use this audit to distinguish original-set fitting from final-selection errors before the next model change.',
        'A better original-set MSE alone is neither final-selection calibration nor deployability.',
        'No new inferential CI is claimed for these descriptive conditional diagnostics;primary policy intervals',
        'are reported in the corresponding whole-roster results,including negative and unknown-completion outcomes.',
        'Exposed SOURCE fitting only,not whole-pipeline OOF or independent confirmation.',
        'Task:obs8/pred12 at raw stride12,detector-silver/local-image coordinates.',
        'No new trajectory dynamics,metric,seconds,human-gold,true3D,foundation,submission-ready claim.',
        'Deployment unchanged;independent roles,Stage5C execution and SMC stay closed.', '']
    return '\n'.join(lines)


def main():
    value = build(); p = run.PUBLIC; (p/'readout.md').write_text(render(value))
    run.write(p/'evidence.json', dict(complete_sha256=run.sha(p/'complete.json'),
        summary_sha256=run.sha(p/'summary.json'), verification_sha256=run.sha(p/'verification.json'),
        report_code_sha256=run.sha(Path(__file__)), report_sha256=run.sha(p/'readout.md'),
        no_new_model_or_policy=True, independent_confirmation=False, deployment_changed=False))
    print(json.dumps(dict(all72groups_exact=True,
        all288legacy_MSE_controls_close=value['summary']['all288_original_selected_MSE_controls_close'], new_model_or_policy=False)))


if __name__ == '__main__': main()
