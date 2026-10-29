import copy
import pytest
from scripts import verify_m3w_pooled_analogue as check
from tests.test_m3w_selected_analogue_verifier import example


def fixture():
    rows,_,expected=example();names={v:k for k,v in check.NAMES.items()}
    for row in rows:
        row['scores']={names[k]:v for k,v in row['scores'].items()}
        source=row['source'];ref=row['references'][0]
        ref.update(recording='["'+source+'","query"]',query_source=source,local_recording='query',
                   known_donors_by_source={'s0':8,'s1':8,'s2':8,'s3':8})
    summary=check.run.api.summarize(rows,expected)
    return rows,summary,expected


def test_pooled_arithmetic_and_namespace_replay():
    r=check.verify(*fixture())
    assert r['pooled_bank_attribution_checks']==72 and r['magnitude_comparison_checked']


@pytest.mark.parametrize('mutation',['source','count','magnitude'])
def test_donor_or_comparison_tampering_rejected(mutation):
    rows,summary,expected=fixture();ref=rows[0]['references'][0]
    if mutation=='source':ref['query_source']='wrong'
    if mutation=='count':ref['known_donors_by_source']['s0']=9
    if mutation=='magnitude':summary['magnitude_better_than_origin']=True
    with pytest.raises(ValueError):check.verify(rows,summary,expected)
