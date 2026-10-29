import copy
import json
import numpy as np
import pytest
from scripts import verify_m3w_coupled_support as verifier
from tests.test_m3w_local_cost_analogue import inputs


@pytest.fixture(scope='module')
def evidence():
    run=verifier.run;b,q,p=inputs();parts=[]
    for source in ('a','b','c','d'):
        part=copy.deepcopy(b);part['sources'][:]=source
        part['recordings']=np.array([json.dumps([source,json.loads(r)[1]],separators=(',',':')) for r in b['recordings']])
        parts.append(part)
    bank={k:np.concatenate([v[k] for v in parts]) for k in b};q=np.tile(q,4);p=np.tile(p,(4,1))
    out=run.api.retrieve(bank,q,p);rows=[]
    for source in ('a','b','c','d'):
        take=bank['sources'][out['indices']]==source;ix=out['indices'][take]
        pred={'origin':p[ix].copy()}
        for arm in run.api.MODES:
            pred[arm]=p[ix].copy();pred[arm][:,[1,4]]=out['predictions'][arm][take]
        score={a:run.api.base.score(bank['labels'][ix],v,bank['envelope'][ix],2.,out['supported'][take]) for a,v in pred.items()}
        rows.append(dict(source=source,scores=score,diagnostics=run.diagnostic(out,take,bank['labels'][ix],pred,bank['envelope'][ix],2.),
            references=[r for r in out['references'] if r['query_source']==source],prior_uniform_MSE={'all':1.,'easy':1.}))
    full=[dict(copy.deepcopy(r),context=str(c),seed=s) for c in range(6) for s in range(3) for r in rows]
    expected={(r['context'],r['source'],r['seed']) for r in full}
    return full,run.api.summarize(full,expected),expected


def test_separate_verifier_accepts_arithmetic(evidence):
    rows,summary,expected=evidence
    assert verifier.verify(rows,summary,expected)['arm_targets']==864


@pytest.mark.parametrize('fault',['cohort','moment','overlap','slice','alpha','gate'])
def test_verifier_detects_inconsistent_receipts(evidence,fault):
    rows,summary,expected=copy.deepcopy(evidence)
    if fault=='cohort':rows[0]['scores']['nested_support_blend']['unknown']+=1
    if fault=='moment':rows[0]['diagnostics']['arms']['coupled_cost']['easy_exceeds_all']=1
    if fault=='overlap':rows[0]['diagnostics']['overlap']['inconsistent_True_low_record_True']+=1
    if fault=='slice':rows[0]['diagnostics']['slices']['alpha_zero']['signed_SSE']['origin']['easy']+=10
    if fault=='alpha':rows[0]['diagnostics']['alpha_sum']=9999
    if fault=='gate':summary['magnitude_screen_pass']=not summary['magnitude_screen_pass']
    with pytest.raises((ValueError,AssertionError)):verifier.verify(rows,summary,expected)
