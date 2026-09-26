import numpy as np
from scripts.run_m3w_european_cap_exceedance import contrasts


def rows():
    result = []
    for seed in [17, 29, 43]:
        for i in range(4):
            linear = dict(status='measured', BCE=.4, prior_BCE=.5, Brier=.2,
                          prior_Brier=.25, AP=.2, top10_overshoot_mass=.3)
            mlp = dict(linear, BCE=.3, Brier=.15, AP=.3, top10_overshoot_mass=.4)
            result.append(dict(producer=0, controller=1, pair='full', seed=seed,
                               outer=str(i), metrics=dict(linear=linear, mlp=mlp,
                               envelope=dict(AP=.1, top10_overshoot_mass=.2),
                               frozen_easy_fraction=dict(AP=.15), frozen_harm_fraction=dict(AP=.18))))
    return result


def test_paired_contrast_orientation_and_locality_counts():
    cfg = dict(pairs=['full'], arms=['linear', 'mlp'], seeds=[17, 29, 43],
               bootstrap=3000, bootstrap_seed=9)
    result = contrasts(rows(), cfg)
    m = {r['contrast']: r for r in result if r['arm'] == 'mlp'}
    np.testing.assert_allclose(m['BCE_gain_prior']['point'], .2)
    np.testing.assert_allclose(m['BCE_gain_linear']['point'], .1)
    assert m['AP_gain_envelope']['sign'] == 'positive'
    assert m['BCE_gain_prior']['localities'] == 4
    data = rows(); data[0]['metrics']['mlp']['AP'] = None
    result = contrasts(data, cfg)
    missing = next(r for r in result if r['arm'] == 'mlp' and r['contrast'] == 'AP_gain_envelope')
    assert missing['status'] == 'not_estimable'
