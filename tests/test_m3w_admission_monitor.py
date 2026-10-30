import pytest
import torch

from src.world_model import m3w_admission_monitor as api
from src.world_model import m3w_admission_cost as frozen
from tests.test_m3w_admission_cost import example
from tests.test_m3w_active_bank_cost import packets_and_data
from tests.test_m3w_budget_score_cost import settings


def strip(state):
    result = dict(state)
    result.pop('admission_monitor')
    return result


def test_channel_decomposition_equals_actual_frozen_penalty_and_unknowns_are_masked():
    p, y, real = example()
    parts, info = api.parts(p, y, real, torch.ones(5), underharm_weight=4., ranking_weight=.5)
    value, count = frozen.admission_penalty(p, y, real, torch.ones(5))
    torch.testing.assert_close(parts['aux'], value, rtol=0, atol=0)
    torch.testing.assert_close(parts['aux_all'] + parts['aux_easy'], value)
    assert info['false_all_rows'] == info['false_easy_rows'] == 1
    assert info['selected_rows'] == count['selected_supervised_rows'] == 2
    parts['aux_easy'].backward()
    assert not torch.count_nonzero(p.grad[2])
    assert p.grad[0, 6, 4] < 0


@pytest.mark.parametrize('arm', ['upper_score', 'admitted_score'])
def test_instrumented_real_torch_training_keeps_all_scientific_state_exact(tmp_path, arm):
    data, _, _ = packets_and_data()
    kw = dict(objective_arm=arm, variant='new_neural', settings=settings(), seed=17,
              identity=dict(fixture=True))
    reference = frozen.fit(data, **kw, path=tmp_path / 'reference.gz')
    observed = api.fit(data, **kw, path=tmp_path / 'observed.gz')
    frozen.parent.active.assert_scientific_equal(reference, strip(observed))
    monitor = observed['admission_monitor']
    assert monitor['step'] == 6 and len(monitor['rows']) == 4
    assert monitor['draw_hash'] == observed['draw_hash']
    assert monitor['row_draws'] == observed['row_draws']
    assert monitor['unknown_supervised_draws'] == 0
    assert observed['parameters'] == 30405


def test_instrumented_checkpoint_resume_keeps_telemetry_and_old_weights_exact(tmp_path):
    data, _, _ = packets_and_data()
    kw = dict(objective_arm='admitted_score', variant='new_neural', settings=settings(), seed=17,
              identity=dict(fixture=True))
    direct = api.fit(data, **kw, path=tmp_path / 'direct.gz')
    api.fit(data, **kw, path=tmp_path / 'resume.gz', stop_at=3)
    resumed = api.fit(data, **kw, path=tmp_path / 'resume.gz', resume=True)
    frozen.parent.active.assert_scientific_equal(strip(direct), strip(resumed))
    frozen.parent.active.completion.base.exact(direct['admission_monitor'], resumed['admission_monitor'])


def test_all_monkeypatches_restore_on_exception():
    base, detector = frozen.parent.active.completion.base, frozen.parent.active.detector
    before = (base.continuous_loss, base.draw_batch, base.save_checkpoint, detector.DetectorCostHead)
    monitor = api.Monitor('admitted_score', settings())
    with pytest.raises(RuntimeError):
        with api.installed(monitor):
            raise RuntimeError('restore instrumentation on failure')
    after = (base.continuous_loss, base.draw_batch, base.save_checkpoint, detector.DetectorCostHead)
    assert before == after
