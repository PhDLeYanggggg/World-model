"""Inspect rejected primal solutions using fixed causal failures, never labels."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import src.world_model.m3w_interaction_controls as controls
import src.world_model.m3w_easy_allocation as allocator
from scripts.run_m3w_easy_allocation import load, causal_view, query, CONTEXT_KEYS
from scripts.run_m3w_native_forecast import immutable_json, file_digest


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    pack = load(); cfg, data, context, *_ = pack
    source = ROOT/cfg['reports']/'solver_diagnosis.json'
    cases = json.loads(source.read_text())['records']
    records = []; current = None
    for case in cases:
        key, action = case['view'], case['action']
        if current != (key, action):
            values, prediction, scale, cut = causal_view(pack, key, action)
            index = np.full(len(data['sites']), -1, np.int64)
            index[values['ids']] = np.arange(len(values['ids']))
            current = key, action
        rec = next(r for r in context['records'] if r['recording'] == case['recording'])
        with np.load(ROOT/rec['cache']['path'], allow_pickle=False) as z:
            c = {k: z[k].copy() for k in CONTEXT_KEYS}
        original_milp, original_solve = controls.milp, allocator.solve_control
        calls, results = [], []
        def observe_milp(**kwargs):
            result = original_milp(**kwargs)
            results.append(result)
            return result
        def observe_solve(p, **kwargs):
            result = original_solve(p, **kwargs)
            proposed = np.asarray(results[-1].x)
            n = len(p.supported); bits = proposed[:n] > .5
            edges = p.edges if kwargs['objective_kind'] == 'joint' else np.empty((0, 2), int)
            products = bits[edges[:, 0]] & bits[edges[:, 1]]
            parts = controls.decompose_pair_objective(p)
            desc = controls._describe(p, bits, parts)
            calls.append(dict(objective=kwargs['objective_kind'], reason=result['reason'],
                proposed_count=int(bits.sum()), requested_count=kwargs.get('exact_interventions'),
                integer_error=float(np.max(np.abs(proposed[:n]-bits))),
                product_error=float(np.max(np.abs(proposed[n:]-products))) if len(edges) else 0.,
                unsupported_selected=int((bits & ~p.supported).sum()),
                risk=desc['mean_predicted_harm'], budget=p.max_mean_predicted_harm,
                risk_overrun=desc['mean_predicted_harm']-p.max_mean_predicted_harm,
                predicted_constraints_satisfied=desc['predicted_constraints_satisfied']))
            return result
        controls.milp, allocator.solve_control = observe_milp, observe_solve
        try:
            query(np.flatnonzero(c['context_frame_ids'] == case['frame']), c, index,
                  values, prediction, data, scale, cut, action, cfg)
        finally:
            controls.milp, allocator.solve_control = original_milp, original_solve
        records.append({**{k: case[k] for k in ('view','action','recording','frame')}, 'calls': calls})
    rejected = [c for r in records for c in r['calls'] if c['reason'] == 'solver_solution_invalid_floor']
    report = dict(result_source='fresh_causal_primal_diagnosis', cases=records,
        rejected_calls=len(rejected),
        rejected_with_risk_overrun=sum(c['risk_overrun'] > 1e-10 for c in rejected),
        max_integer_error=max(c['integer_error'] for c in rejected),
        max_product_error=max(c['product_error'] for c in rejected),
        future_outcome_arrays_loaded=False, policy_changed=False,
        source_sha256=file_digest(source), code_sha256=file_digest(Path(__file__)))
    immutable_json(ROOT/cfg['reports']/'primal_diagnosis.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
