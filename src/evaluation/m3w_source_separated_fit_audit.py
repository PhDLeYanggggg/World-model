"""Read-only fitting isolation checks; no target or validation file is read."""
import numpy as np

from src.data_unification.m3w_moving_forecast_data import array_hash


def audit_state(state, receipt, part, seed):
    expected = part['routing_job']['evaluation_scope']
    names = set(part['names'])
    pr = state['preprocess']
    if (set(state['source_sites']) != names or set(pr['source_names']) != names
            or set(receipt['fit_sources']) != names
            or len(state['fit_ids']) != expected['rows'] or pr['fitting_rows'] != expected['rows']
            or array_hash(state['fit_ids']) != expected['ids_sha256']
            or receipt['fit_ids_sha256'] != expected['ids_sha256']):
        raise ValueError('Checkpoint, normalizer and receipt must use only their exact own fitting rows')
    if (state['step'] != 2000 or state['budget'] != 2000 or state['seed'] != seed
            or state['parameters'] != 28229 or int(state['sampler']['counts'].min()) < 4
            or state['unknown_rows_mislabeled_as_observed'] != 0):
        raise ValueError('Complete real fixed-budget neural fit required')
    if any(array_hash(pr[k]) != receipt['preprocess_array_hashes'][k] for k in ('mean', 'std', 'rms', 'bias')):
        raise ValueError('Saved fitting normalizer changed')
    if not all(np.isfinite(row['total']) for row in state['trace']) or state['trace'][-1]['step'] != 2000:
        raise ValueError('Finite complete fitting trace required')
    if not state['optimizer']['state'] or any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
        raise ValueError('Every optimized parameter must have2000actual Adam updates')
    moved = sum(int(np.any(v.detach().cpu().numpy() != state['initial_model'][k].detach().cpu().numpy()))
        for k, v in state['model'].items())
    if not moved:
        raise ValueError('A checkpoint with no changed neural parameter is not trained')
    return dict(own_source_and_row_scope_exact=True, normalizer_scope_exact=True,
        optimizer_updates_each_parameter=2000, changed_parameter_tensors=moved,
        finite_complete_training_trace=True, no_validation_labels_read=True)
