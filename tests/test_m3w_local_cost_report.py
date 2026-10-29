from scripts import report_m3w_local_cost_analogue as report
from tests.test_m3w_local_cost_verifier import fixture


def test_report_preserves_undefined_views_and_violations():
    rows,_,_=fixture()
    for r in rows:
        for a in report.run.api.ARMS:
            s=r['scores'][a];s['benefit_harm_envelope_exceedances']=1
            r['diagnostics']['arms'][a]['easy_exceeds_all']=2
            for t in ('all','easy'):
                s[t].update(observed_harm=1.,false_safe_rows=1,AUROC=.5,AP=.4)
    rows[0]['scores']['local_signed32']['easy']['AUROC']=None
    data,md=report.describe(rows)
    assert data['all']['local_signed32']['easy']['AUROC'] is None
    assert data['all']['local_signed32']['easy_exceeds_all']==144
    assert data['all']['local_signed32']['benefit_harm_envelope_exceedances']==72
    assert len(data['source'])==4 and len(data['context'])==6 and len(data['seed'])==3
    assert 'not overall query median' in md and 'not evidence of prediction lift' in md
