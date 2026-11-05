"""Synthetic orchestration checks, not evidence of real-data safety."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.evaluation import m3w_fixed_risk_readout as reader


def test_all_model_decisions_precede_outcomes_and_replay_is_exact(tmp_path, monkeypatch):
    source = reader.source
    monkeypatch.setattr(source, 'ROOT', tmp_path)
    events = []
    ids = np.array([101, 102])
    causal = {'ids': ids}
    policy = dict(known_ADE_gain_floor_percent=0., switch_rate=0., subsets={'hard': {'gain_percent': 0.}})
    cached = tmp_path/'control.npz'
    zero = np.zeros(2, dtype=np.int64)
    np.savez_compressed(cached, ids=ids, constant_veto=zero, constant_matched=zero)
    old_row = dict(cache=source.artifact(cached), choice_hashes={
        k: source.array_hash(zero) for k in ('constant_veto', 'constant_matched')},
        rows=[dict(site='open_source', policies={k: policy for k in ('constant_veto', 'constant_matched')})])
    monkeypatch.setattr(source, 'read', lambda path: old_row)
    packet = dict(causal=causal, ids=ids, sites=np.array(['open_source']*2),
        paths=np.zeros((2, 7, 12, 2)), floor=np.zeros((2, 12, 2)), cv=np.zeros((2, 12, 2)),
        recordings=np.array(['record']*2), frames=np.array([0, 0]))
    monkeypatch.setattr(reader.old, 'packet', lambda *args, **kwargs: packet)
    monkeypatch.setattr(reader.direct, 'decisions', lambda *args, **kwargs:
        (None, dict(router=zero, all_action=zero), None))
    monkeypatch.setattr(reader.old, 'rollout', lambda paths, floor, choice: floor.copy())
    monkeypatch.setattr(reader.inherited, 'assert_excluded', lambda *args: None)

    def choose(*args):
        events.append('choose')
        return zero.copy(), zero.copy()

    monkeypatch.setattr(reader.inherited.api.kin.parent, 'choose', choose)
    monkeypatch.setattr(source, 'readout', SimpleNamespace(
        policy=lambda *args, **kwargs: policy,
        paired=lambda *args: dict(known_pp=0., completion_lower_pp=0., completion_upper_pp=0.)))
    monkeypatch.setattr(source, 'matched_utility', lambda *args: dict(status='not_estimable'))
    monkeypatch.setattr(reader.direct, 'safety', lambda *args:
        dict(full_original_risk_pass=False, supported=0, required=1))
    monkeypatch.setattr(reader.inherited, 'macro', lambda rows, fn: dict(mean=float(np.mean([fn(r) for r in rows]))))

    class Outcomes(dict):
        def __getitem__(self, key):
            if key in ('target_eval', 'valid'):
                assert events.count('predict') == 2
                assert events.count('choose') == 4
                events.append('outcome')
            return super().__getitem__(key)

    data = Outcomes(target_eval=np.zeros((103, 12, 2)), origin=np.zeros((103, 2)),
        valid=np.ones((103, 12), dtype=bool))
    jobs = [dict(key='fit', easy_cut=1., hard_cut=2.)]
    values = (None, None, data, jobs, None, {'fit': [None, None, {'names': ['held']}]}, None, None, None, None)
    def predict(state, c):
        assert c is causal and 'outcome' not in events
        events.append('predict')
        return np.zeros((2, 7, 2)), np.zeros((2, 7, 2))
    kw = dict(names=('independent', 'composed'), control_public=tmp_path/'control',
        public=tmp_path/'public', private=tmp_path/'private', guard=lambda: None, beat=lambda **kw: None)
    models = {('fit', name): {} for name in kw['names']}
    result = reader.evaluate(values, models, predict, **kw)
    assert not result['held_policy']['composed_veto']['safety']['full_original_risk_pass']
    events.clear()
    assert reader.evaluate(values, models, predict, **kw, verify=True) == result
    cache = tmp_path/'private/evaluation/fit_val2_propose1.npz'
    with np.load(cache, allow_pickle=False) as saved:
        contents = {k: saved[k].copy() for k in saved.files}
    contents['composed_risk_prediction'][0, 0, 0] = 1.
    np.savez_compressed(cache, **contents)
    events.clear()
    with pytest.raises(AssertionError):
        reader.evaluate(values, models, predict, **kw, verify=True)
