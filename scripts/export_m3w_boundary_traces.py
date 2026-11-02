"""Export scalar training monitors from hash-verified completed private weights."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_boundary_guard as run


def main():
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    cells = []
    for ref in freeze['fits']:
        if run.sha(run.ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Registered complete fit receipt changed')
        r = json.loads((run.ROOT/ref['path']).read_text())
        path = run.ROOT/r['checkpoint']['path']
        if run.sha(path) != r['checkpoint']['sha256']:
            raise ValueError('Completed training checkpoint changed')
        state = run.api.parent.parent.read_checkpoint(path)
        if state['trace'][0] != r['first_trace'] or state['trace'][-1] != r['final_trace']:
            raise ValueError('Exported trace must agree with frozen endpoints')
        trace = []
        for v in state['trace']:
            m = v['monitor']
            trace.append(dict(step=v['step'], anchor_monitor=m['anchor_objective'],
                boundary_monitor=m['boundary_objective'], total_monitor=m['total'],
                gradient_norm=v.get('gradient_norm'), minimum_query_visits=v.get('minimum_query_visits')))
        cells.append(dict(context=r['context'], seed=r['seed'], arm=r['arm'], budget=r['budget'],
            checkpoint_sha256=r['checkpoint']['sha256'], trace=trace))
    if len(cells) != 36:
        raise ValueError('All 36 complete traces required')
    run.once(run.PUBLIC/'training_traces.json', dict(result_source='fresh_scalar_export_of_cached_verified_real_training_checkpoints',
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'), cells=cells,
        monitor_role='fixed_fitting_queries_not_validation', optimization_reexecuted=False,
        raw_data_features_or_weights_included=False))


if __name__ == '__main__': main()
