"""Fitting-state conditioning diagnosis; no held labels or optimizer update."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_record_policy as run


def main():
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    rows = []
    for ref in freeze['fits']:
        path = ROOT/ref['path']; receipt = json.loads(path.read_text())
        if run.sha(path) != ref['sha256'] or run.sha(ROOT/receipt['checkpoint']['path']) != receipt['checkpoint']['sha256']:
            raise ValueError('Frozen fitting state changed')
        state = run.api.read_checkpoint(ROOT/receipt['checkpoint']['path'])
        ratio = (state['minimum_real_easy_support']/state['record_risk_scales'][:, 1]).numpy()
        maximum = np.minimum(state['settings']['dual_maximum'], state['step']*state['settings']['dual_learning_rate']*ratio)
        actual = state['dual'][:, 2].numpy()
        if (actual > maximum+1e-5+maximum*1e-5).any():
            raise ValueError('Expected upper bound on original support dual does not hold')
        rows.append(dict(context=receipt['context'], seed=receipt['seed'], arm=receipt['arm'],
            records=len(ratio), empty_support_excess=ratio.tolist(), maximum_support_dual=maximum.tolist(),
            actual_support_dual=actual.tolist(), final_trace_gradient_norm=receipt['final_trace']['gradient_norm'],
            final_trace_selected_rows=receipt['final_trace']['selected_rows']))
    summary = {}
    for arm in run.api.ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        ratio = np.concatenate([r['empty_support_excess'] for r in selected])
        actual = np.concatenate([r['actual_support_dual'] for r in selected])
        maximum = np.concatenate([r['maximum_support_dual'] for r in selected])
        summary[arm] = dict(record_head_views=len(ratio),
            empty_support_excess_quantiles=np.quantile(ratio, [0,.25,.5,.75,1]).tolist(),
            actual_support_dual_quantiles=np.quantile(actual, [0,.25,.5,.75,1]).tolist(),
            maximum_dual_below_one=int((maximum < 1).sum()),
            zero_selected_endpoints=sum(r['final_trace_selected_rows'] == 0 for r in selected))
    result = dict(result_source='fresh_run_frozen_fitting_state_numeric_diagnosis', summary=summary, rows=rows,
        original_training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        code_sha256=run.sha(Path(__file__)),
        support_dual_growth_upper_bound_verified=True,
        candidate_repair='scale_positive_support_by_its_own_minimum_not_full_easy_reference',
        repair_preserves_constraint_sign_and_evaluation_budget=True,
        no_optimizer_update=True, held_labels_read=False, independent_roles_opened=False)
    run.once(run.PUBLIC/'support_unit_diagnosis.json', result)
    print(json.dumps(dict(summary=summary, result_sha256=run.sha(run.PUBLIC/'support_unit_diagnosis.json'))))


if __name__ == '__main__':
    main()
