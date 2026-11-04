import ast
from pathlib import Path
import numpy as np
import pytest

from scripts import run_m3w_moving_source_separated_critics as run


def test_training_definition_has_no_H_or_rule_change():
    spec = run.source.read(run.CONFIG)
    assert spec['new_neural_cost_heads'] == 18*3*2
    assert spec['primary_optimizer_updates'] == 2000*108
    assert spec['positive_harm_budget'] == spec['easy_net_degradation_budget'] == .02
    assert not spec['analytic_identity_certification_added']
    assert not spec['new_H_prediction_or_policy_scoring']
    tree = ast.parse(Path(run.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert 'held' not in ast.unparse(main)


def test_causal_packet_does_not_index_future_labels_or_fit_preprocessing(monkeypatch):
    class Forbidden:
        def __getitem__(self, key): raise AssertionError('Inference read a future target')
    data = dict(geometry=np.zeros((2, 476)), shared_context=np.zeros((2, 48)),
        target_eval=Forbidden(), valid=Forbidden(), origin=Forbidden())
    raw = dict(ids=np.arange(2), sites=np.array(['a', 'a']), causal={}, paths=np.zeros((2, 7, 12, 2)),
        recordings=np.zeros(2, int), frames=np.zeros(2, int))
    def view(*args, **kwargs):
        assert kwargs == dict(training=False)
        return raw
    monkeypatch.setattr(run.source, 'view', view)
    monkeypatch.setattr(run.amplitude, 'causal_bank', lambda **_: dict(causal={}, paths=raw['paths']))
    job = dict(producer_sites=['P'], evaluation_sites=['H'], floor_index=1)
    p = run.packet(data, job, {}, dict(names=['a'], routing_job={}), training=False)
    assert 'learning' not in p


def test_training_packet_rejects_full_C_normalizer(monkeypatch):
    raw = dict(ids=np.arange(2), sites=np.array(['a', 'a']), causal={}, paths=np.zeros((2, 7, 12, 2)),
        recordings=np.zeros(2, int), frames=np.zeros(2, int))
    monkeypatch.setattr(run.source, 'view', lambda *a, **k: raw)
    monkeypatch.setattr(run.amplitude, 'causal_bank', lambda **_: {})
    monkeypatch.setattr(run.source.api.base, 'training_data', lambda *a: dict(preprocess=dict(source_names=['a', 'b'])))
    data = dict(geometry=np.zeros((2, 476)), shared_context=np.zeros((2, 48)),
        target_eval=np.zeros((2, 12, 2)), valid=np.ones((2, 12), bool), origin=np.zeros((2, 2)))
    with pytest.raises(ValueError, match='Own single-part'):
        run.packet(data, dict(producer_sites=['P'], evaluation_sites=['H'], floor_index=1, easy_cut=1.), {},
            dict(names=['a'], routing_job={}), training=True)
