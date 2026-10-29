import copy
import json
import pytest
from scripts import verify_m3w_local_cost_analogue as verify


def fixture():
    rows=[]
    for context in range(6):
        for source in range(4):
            for seed in (17,29,43):
                scores={};arms={}
                for a in verify.run.api.ARMS:
                    m=.5 if a=='local_signed32' else 1.
                    targets={t:dict(full_cohort_MSE=m,events=1,predicted_harm=2.) for t in ('all','easy')}
                    scores[a]=dict(rows=4,known=3,unknown=1,supported=4,unsupported=0,
                                   supported_known=3,supported_unknown=1,**targets)
                    arms[a]={t:dict(events=1,event_harm_SSE=1.,non_event_harm_SSE=1.,
                                   predicted_harm_on_events=1.,predicted_harm_on_non_events=1.) for t in ('all','easy')}
                ref=dict(recording=json.dumps([str(source),'r']),query_source=str(source),local_recording='r',
                    query_rows=4,same_recording_donors=0,unselected_donors=0,known_donors=32,
                    known_donors_by_source={'a':32},supported=True)
                rows.append(dict(context=context,source=str(source),seed=seed,scores=scores,references=[ref],
                    diagnostics=dict(arms=arms,supported_queries=4,known_queries=3,effective_donor_quantiles=[32.]*5,
                        max_weight_quantiles=[1/32]*5,donor_loss={'uniform':[2.,2.],'optimum':[1.,1.]})))
    expected=[(r['context'],r['source'],r['seed']) for r in rows]
    summary=verify.run.api.summarize(rows,expected)
    return rows,summary,expected


def test_valid_complete_receipts():
    r,s,e=fixture();out=verify.verify(r,s,e)
    assert out['arm_target_checks']==864 and out['magnitude_screen_pass']


@pytest.mark.parametrize('change',('cohort','donor','support','loss','summary','leverage','drop'))
def test_corruption_detected(change):
    rows,s,e=fixture();r=rows[0]
    if change=='cohort':r['scores']['local_signed32']['unknown']=0
    if change=='donor':r['references'][0]['unselected_donors']=1
    if change=='support':r['scores']['local_signed32']['unsupported']=1
    if change=='loss':r['diagnostics']['donor_loss']['optimum']=[3.,3.]
    if change=='summary':s['pair_wins']=17
    if change=='leverage':r['diagnostics']['effective_donor_quantiles']=[33.]*5
    if change=='drop':rows.pop()
    with pytest.raises((AssertionError,ValueError)):verify.verify(rows,s,e)
