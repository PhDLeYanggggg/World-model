import numpy as np

from src import stage41_fixed_prior_oracle_audit as oracle


def test_oracle_chosen_selects_lowest_cost_source(monkeypatch):
    pack = {"labels": {}, "horizon": np.asarray([50, 100])}
    costs = {
        "bridge": np.asarray([3.0, 1.0]),
        "old_shape": np.asarray([2.0, 4.0]),
        "gain_gate": np.asarray([5.0, 0.5]),
    }

    monkeypatch.setattr(oracle, "_source_cost_matrix", lambda _pack: np.column_stack([costs[source] for source in oracle.SOURCES]))

    chosen = oracle._oracle_chosen(pack)
    assert chosen.tolist() == [oracle.SOURCES.index("old_shape"), oracle.SOURCES.index("gain_gate")]


def test_delta_reports_oracle_minus_fixed():
    compact = {
        "all": 1.5,
        "t50": 2.0,
        "t100": 3.0,
        "hard_failure": 4.0,
        "easy_degradation": 0.0,
        "shape_gain_all": 0.4,
        "shape_gain_t50": 0.3,
        "shape_gain_t100": 0.2,
        "shape_gain_hard_failure": 0.1,
    }
    fixed = {key: 1.0 for key in compact}
    fixed["easy_degradation"] = 0.0

    delta = oracle._delta(compact, fixed)

    assert delta["all"] == 0.5
    assert delta["t50"] == 1.0
    assert delta["easy_degradation"] == 0.0
