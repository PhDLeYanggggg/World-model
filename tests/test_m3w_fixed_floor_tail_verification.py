import numpy as np
from scripts.verify_m3w_fixed_floor_tail import independently_match,reduce_check


def test_count_reconstruction_never_borrows_later_query():
    result,queries=independently_match([True,False,False,False],[True]*4,[9.,1.,0.,0.],
        ['a']*4,[1,1,2,2],[1,2,3,4])
    assert result.tolist()==[False,True,False,False] and queries==2


def test_tie_break_is_row_id_not_array_position():
    result,_=independently_match([True,False],[True,True],[0.,0.],['a','a'],[1,1],[9,2])
    assert result.tolist()==[False,True]


def test_undefined_source_not_dropped_from_primary():
    r=dict(by_site={'a':1.,'b':None},point=None,ci95=None)
    rows=[dict(site='a',metric={'x':1.}),dict(site='b',metric={'x':None})]
    assert reduce_check(r,rows,'x',17,3000)==1
