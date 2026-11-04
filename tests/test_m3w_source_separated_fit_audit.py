import copy
import numpy as np
import pytest
import torch

from src.evaluation.m3w_source_separated_fit_audit import audit_state
from src.data_unification.m3w_moving_forecast_data import array_hash


def fixture():
    ids = np.array([1, 4])
    normalizer = dict(source_names=['a'], fitting_rows=2, **{k: np.ones(2) for k in ('mean', 'std', 'rms', 'bias')})
    state = dict(preprocess=normalizer, source_sites=['a'], fit_ids=ids, step=2000, budget=2000, seed=17,
        parameters=28229, sampler=dict(counts=np.array([4, 8])), unknown_rows_mislabeled_as_observed=0,
        trace=[dict(step=2000, total=.1)], optimizer=dict(state={0: dict(step=torch.tensor(2000))}),
        model=dict(w=torch.ones(2)), initial_model=dict(w=torch.zeros(2)))
    receipt = dict(fit_sources=['a'], fit_ids_sha256=array_hash(ids),
        preprocess_array_hashes={k: array_hash(normalizer[k]) for k in ('mean', 'std', 'rms', 'bias')})
    part = dict(names=['a'], routing_job=dict(evaluation_scope=dict(rows=2, ids_sha256=array_hash(ids))))
    return state, receipt, part


def test_actual_optimizer_state_and_own_scope_required():
    s, r, p = fixture()
    assert audit_state(s, r, p, 17)['no_validation_labels_read']
    for mutation in ('rows', 'normalizer', 'optimizer', 'unchanged'):
        bad = copy.deepcopy(s)
        if mutation == 'rows': bad['fit_ids'][0] = 9
        if mutation == 'normalizer': bad['preprocess']['source_names'].append('V')
        if mutation == 'optimizer': bad['optimizer']['state'][0]['step'] = torch.tensor(1999)
        if mutation == 'unchanged': bad['model'] = bad['initial_model']
        with pytest.raises(ValueError): audit_state(bad, r, p, 17)
