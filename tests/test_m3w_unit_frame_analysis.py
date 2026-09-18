import pytest

from scripts.analyze_m3w_unit_frame_training import gain, summarize_arm


def test_zero_reference_is_undefined_not_infinite_success():
    assert gain([1.,2.],[0.,0.]) is None
    assert gain([1.,2.],[2.,4.]) == 50.


def test_analysis_preserves_harm_and_uses_equal_cells():
    def trial(rows,error,ref):
        metric=dict(rows=rows,primary_ADE=error,reference_ADE=ref,
            mean_harm_over_reference=error-ref,easy_degradation_percent=100*(error/ref-1),
            easy_absolute_harm=error-ref,improvement_percent=100*(1-error/ref),forecast_nonfinite=0)
        return dict(slices={'event':dict(metric)},vs_CV=metric,train_equal_scene_gain_percent=4.)
    summary=summarize_arm([trial(1000,1.,2.),trial(1,10.,5.)])
    assert summary['slices']['event']['gain_percent'] == pytest.approx(100*(1-11/7))
    assert summary['easy_absolute_harm_range']==[-1.,5.]
    assert summary['finite_prediction_failures']==0
