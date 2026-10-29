import copy
from tests.test_m3w_pooled_analogue_verifier import fixture
from scripts import report_m3w_pooled_analogue as report


def test_report_preserves_full_roster_and_labels_subset_diagnostic():
    rows,_,_=fixture()
    old={}
    for row in rows:
        previous=copy.deepcopy(row)
        previous['scores']['nearest32']=previous['scores']['pooled_nearest32']
        if row['source']!='s0':previous['scores']['nearest32']['unsupported']=3
        old[row['context'],row['source'],row['seed']]=previous
    result,table=report.describe(rows,old)
    assert result['all']['pooled_nearest32']['rows']==216
    assert result['all']['pooled_nearest32']['unknown']==72
    assert result['previous_complete_support_views']==18
    assert result['previous_complete_support_current']['pooled_nearest32']['rows']==54
    assert not result['screen_changed'] and table.count('| source:')==4
