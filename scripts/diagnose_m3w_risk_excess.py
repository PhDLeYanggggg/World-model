"""Post-freeze worst-view and zero-reference checks; never policy fitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_excess as run
import numpy as np


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    _, data, jobs, old_identity, _ = run.load()
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    out = dict(summary=run.artifact(run.PUBLIC/'summary.json'), threshold_changes=False,
               model_changes=False, deployment_changed=False, candidates={}, zero_reference_cases=[])
    for arm in ('dimensionless', 'damped'):
        rr = [r for r in summary['rows'] if r['candidate'] == arm]
        worst = min(rr, key=lambda r: r['metric']['held_new_easy_ADE_gain_vs_CV_percent'])
        out['candidates'][arm] = dict(worst_easy_view=worst['group'], worst_easy_site=worst['site'],
            worst_easy_gain_percent=worst['metric']['held_new_easy_ADE_gain_vs_CV_percent'],
            view_count=len(rr), views_easy_degradation_above2=sum(r['metric']['held_new_easy_ADE_gain_vs_CV_percent'] < -2 for r in rr),
            zero_reference_harmed_views=sum(r['metric']['held_new_zero_reference_harmed'] for r in rr))
    affected = {r['group']: r for r in summary['rows'] if r['metric']['held_new_zero_reference_harmed'] > 0}
    for c in run.parent.contexts(data, jobs, old_identity):
        prefix = run.parent.parent.group_name(c['job'], c['controller'], c['candidate'])+'_held'
        groups = [k for k in affected if k.startswith(prefix)]
        for name in groups:
            ids = c['ids']; site = affected[name]['site']
            with np.load(run.PRIVATE/'heads'/name/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids); selected = z['score'] <= 0
            zero = selected & (data['sites'][ids] == site) & (data['baseline_ade'][ids, 1] == 0)
            glob = ids[zero]; local = np.flatnonzero(zero)
            err = run.parent.parent.native_errors(c['prediction'][local].astype(float)+data['origin'][glob, None],
                data['target_eval'][glob], data['valid'][glob], np.ones(len(glob)))[0]
            bad = err > 0
            moving = np.linalg.norm(data['history'][glob, -1]-data['history'][glob, -2], axis=1) > 0
            assert int(bad.sum()) == affected[name]['metric']['held_new_zero_reference_harmed']
            out['zero_reference_cases'].append(dict(group=name, site=site, selected_zero_rows=len(glob),
                positive_harm_rows=int(bad.sum()), positive_harm=float(err[bad].sum()),
                stationary_last_step_harmed=int((bad & ~moving).sum()),
                moving_last_step_harmed=int((bad & moving).sum())))
            del affected[name]
        if not affected: break
    assert not affected
    run.immutable_json(run.PUBLIC/'slice_diagnosis.json', out)
    lines = ['# Post-Freeze Worst-View Diagnosis', '',
        'Descriptive checks only; no threshold, model, feature or frozen action changes.',
        'Locality-mean easy gates do not assert that every seed/producer view is safe.', '',
        '| Candidate | Worst easy gain % | View | Views worse than -2% |', '|---|---:|---|---:|']
    for a, r in out['candidates'].items():
        lines.append(f"| {a} | {r['worst_easy_gain_percent']:.4f} | {r['worst_easy_view']} | {r['views_easy_degradation_above2']}/{r['view_count']} |")
    lines += ['', '## Reference-Exact Cases', '']
    for r in out['zero_reference_cases']:
        lines.append(f"- {r['group']}: {r['positive_harm_rows']} harmed row-view, absolute image-local ADE harm {r['positive_harm']:.8f}; stationary-last-step {r['stationary_last_step_harmed']}, moving {r['moving_last_step_harmed']}.")
    lines += ['', 'The diagnostic rule intentionally omits the old full policy stationary/utility/easy',
        'guards. A stationary case would be denied by that already existing stationary guard,',
        'but this check does not rerun or certify the complete policy. Do not discard a moving',
        'failure or infer independent safety from an aggregate easy mean. Positive harm at zero',
        'reference has no valid percentage denominator. No deployment change.']
    (run.PUBLIC/'slice_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out['candidates']))


if __name__ == '__main__': main()
