"""Post-readout policy-region diagnosis; no new allocation or future label arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selection_harm_deviance_v1'
CONTROL = ROOT/'outputs/publication_readiness_2026_09/european_selection_cardinal_cost_v1'


def checked(ref):
    p = ROOT/ref['path']
    if hashlib.sha256(p.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Frozen prediction/choice changed')
    return p


def classify(old_choice, new_choice, prediction):
    a, b, p = np.asarray(old_choice), np.asarray(new_choice), np.asarray(prediction)
    if (a.shape != b.shape or a.ndim != 1 or p.shape != (len(a),7,5)
            or not np.isin(a, np.arange(-1,7)).all() or not np.isin(b, np.arange(-1,7)).all()
            or not np.isfinite(p).all() or (p < 0).any()):
        raise ValueError('Aligned frozen choices and native new costs required')
    ix = np.flatnonzero(a >= 0); q = p[ix, a[ix]]
    gain_lost = q[:,0]-q[:,1] <= 0
    guard_lost = ((q[:,2] <= 0) | (q[:,3] <= 0) | (q[:,1] > .02*q[:,2]) | (q[:,4] > .02*q[:,3]))
    counts = dict(old_selected=len(ix), new_selected=int((b >= 0).sum()),
        same_action=int((b[ix] == a[ix]).sum()), now_fallback=int((b[ix] < 0).sum()),
        gain_only=int((gain_lost & ~guard_lost).sum()), guard_only=int((guard_lost & ~gain_lost).sum()),
        both_gain_and_guard=int((gain_lost & guard_lost).sum()), neither_rejected=int((~gain_lost & ~guard_lost).sum()),
        total_rows=len(a))
    if sum(counts[k] for k in ('gain_only','guard_only','both_gain_and_guard','neither_rejected')) != len(ix):
        raise ValueError('Every original intervention must be accounted for')
    return counts


def main():
    read = lambda p:json.loads(Path(p).read_text())
    verification = read(PUBLIC/'verification.json')
    if not verification['all18_new_cost_inferences_exact'] or not verification['all54_metric_groups_exact']:
        raise ValueError('Complete registered readout/replay required')
    cases = []
    for ref in read(PUBLIC/'policy_freeze.json')['policies']:
        r = read(checked(ref)); c = read(CONTROL/'policies'/(r['trial']+'_mean_absolute.json'))
        old = np.load(checked(c['choices']), allow_pickle=False)
        new = np.load(checked(r['choices']), allow_pickle=False)
        pred = np.load(checked(r['prediction']), mmap_mode='r', allow_pickle=False)
        cases.append(dict(trial=r['trial'], arm=r['arm'], counts=classify(old,new,pred)))
    if len(cases) != 18 or len({(r['trial'],r['arm']) for r in cases}) != 18:
        raise ValueError('All18frozen policy comparisons required')
    totals = {arm:{k:sum(r['counts'][k] for r in cases if r['arm'] == arm) for k in cases[0]['counts']}
              for arm in ('easy_deviance','dual_deviance')}
    value = dict(result_source='fresh_post_readout_causal_policy_region_diagnosis', cases=cases, totals=totals,
        code_choice_counts_not_trajectory_equality=True, private_future_labels_read=False,
        new_policy_or_threshold_search=False, calibration_access=False, confirmation_access=False, deployment_changed=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (PUBLIC/'policy_suppression.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    lines = ['# Post-Readout Policy Suppression', '',
        'Only frozen causal predictions and choice arrays are read, not future labels.',
        'This is an added descriptive diagnosis after the registered negative result,',
        'not another operating-point search. Counts repeat nine frozen pipelines and',
        'are not independent targets. Choice equality is not trajectory equality.', '',
        '|Arm|Old selected|New selected|Old action: gain only rejected|Guard only rejected|Both rejected|Neither rejected|Old selected now fallback|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in totals.items():
        lines.append('|'+arm+'|'+'|'.join(str(r[k]) for k in ('old_selected','new_selected','gain_only','guard_only',
            'both_gain_and_guard','neither_rejected','now_fallback'))+'|')
    lines += ['', 'The exclusive categories concern each original selected action under',
        'the NEW learned gain and the NEW unchanged2%predicted guard. They explain',
        'where that action becomes ineligible; they do not identify the causal',
        'training mechanism or prove a retained/replaced choice is beneficial.',
        'Labels or oracle safety never construct a new deployment policy here.', '',
        'Guard-only rejection is the largest exclusive category in this measured',
        'comparison. Preserving the old ranking alone therefore cannot reinstate',
        'most original choices under these same new guards. This statement concerns',
        'old selected actions, not a bound over all alternative bank actions. Any',
        'new comparison must preserve this negative result, freeze its policy before',
        'readout and remain exposed development, not untouched confirmation.', '']
    (PUBLIC/'policy_suppression.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(cases), totals=totals)))


if __name__ == '__main__': main()
