from scripts.report_m3w_rational_harm import interval, render
from src.evaluation.m3w_rational_harm_readout import ARMS, CONTROLS


def test_report_keeps_negative_gate_and_scope():
    group = dict(mean_view_MSE={a: 1. for a in ARMS},
        safety={a: dict(selected=0, known_easy_violations=0, upper_easy_violations=0,
                        empty=72, undefined_easy=72, full_support=0) for a in ARMS},
        contrasts={a+'_'+kind: dict(mean=-1., CI95=[-2., -.1])
                   for a in CONTROLS for kind in ('full', 'matched', 'MSE')})
    text = render(dict(fit=group, held=group, advance=False, failures=['unsupported']),
                  dict(seconds=1.), dict(seconds=1.))
    assert '**False**' in text and 'unsupported' in text
    assert 'not independent' in text and 'NOT ADE/FDE' in text
    assert 'Stage5C/SMC disabled' in text
    assert '| rational |' in text and 'Empty/undefined' in text


def test_undefined_interval_remains_undefined():
    assert interval(dict(CI95=None, mean=None)) == 'undefined'
