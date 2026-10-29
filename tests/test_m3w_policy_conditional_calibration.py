import copy
import numpy as np
import pytest

from src.world_model import m3w_policy_conditional_calibration as m


def packet(n=96):
    p = np.tile([.5, .001, 1., .5, .001], (n, 1))
    y = np.tile([0., .1, 1., 1., .1], (n, 1))
    return dict(prediction=p, targets=y, envelope=np.ones(n), eligible=np.ones(n, bool),
        recordings=np.repeat(['a', 'b', 'c'], n//3))


def bank():
    return {s: packet() for s in ('one', 'two', 'three')}


def test_observed_harm_restricts_raw_policy_without_changing_predictions():
    b=bank(); cal=m.fit(b, 'target'); row=packet(); before=row['prediction'].copy()
    a=m.decide(row['prediction'],row['envelope'],row['eligible'],cal)
    assert a['raw'].all() and a['support_only'].all()
    assert not a['global'].any() and not a['conditional'].any()
    np.testing.assert_array_equal(before,row['prediction'])


def test_target_labels_not_an_inference_argument_or_calibration_source():
    b=bank(); row=packet(); cal=m.fit(b, 'target')
    before=m.decide(row['prediction'],row['envelope'],row['eligible'],cal)
    row['targets'][:]=np.nan
    after=m.decide(row['prediction'],row['envelope'],row['eligible'],cal)
    for k in m.ARMS: np.testing.assert_array_equal(before[k],after[k])
    with pytest.raises(ValueError,match='target must be excluded'):m.fit(b, 'one')


def test_repeating_rows_does_not_overweight_recording_or_source():
    b=bank(); b['one']['targets'][:,[1,4]]=.5
    first=m.fit(b,'target')
    b['one']={k:np.repeat(v,7,axis=0) for k,v in b['one'].items()}
    second=m.fit(b,'target')
    np.testing.assert_allclose(first['table']['global']['offset'],second['table']['global']['offset'])


def test_unknowns_never_imputed_and_empty_cells_veto():
    b=bank()
    for r in b.values():r['targets'][1:]=np.nan
    cal=m.fit(b,'target');row=packet()
    assert cal['table']['global']['known']==3 and cal['table']['global']['unknown']==285
    a=m.decide(row['prediction'],row['envelope'],row['eligible'],cal)
    assert a['raw'].all() and not a['support_only'].any() and not a['conditional'].any()


def test_partial_labels_rejected():
    b=bank();b['one']['targets'][0,1]=np.nan
    with pytest.raises(ValueError,match='all-known'):m.fit(b,'target')


def test_zero_envelope_and_eligibility_never_expand():
    row=packet(); row['eligible'][::2]=False;row['envelope'][::3]=0
    cal=m.fit(bank(),'target'); a=m.decide(row['prediction'],row['envelope'],row['eligible'],cal)
    for action in a.values(): assert not (action & ~row['eligible']).any()


def test_unchanged_budget_required():
    cal=m.fit(bank(),'target');cal['budget']=.03;row=packet()
    with pytest.raises(ValueError,match='Frozen budget'):m.decide(row['prediction'],row['envelope'],row['eligible'],cal)


def contexts():
    return {str(i):[dict(packet=dict(identity=dict(source=s, upstream=dict(context=str(i),
        producer_sites=['p'],controller_sites=['c'],upstream_context=dict(fit_sites=['p'])),
        partition=dict(train_recordings=['f','h'],validation_recordings=['v']))),
        partition=dict(fit_recordings=['f'],held_recordings=['h'])) for s in ['a','b','c2','d']]
        for i in range(6)}


def test_producer_and_recording_exclusion_fail_closed():
    c=contexts();assert len(m.lineage(c))==24
    altered=copy.deepcopy(c);altered['0'][0]['packet']['identity']['upstream']['producer_sites'].append('a')
    with pytest.raises(ValueError,match='producer/controller'):m.lineage(altered)
    altered=copy.deepcopy(c);altered['0'][0]['partition']['held_recordings']=['f']
    with pytest.raises(ValueError,match='recording roles'):m.lineage(altered)
