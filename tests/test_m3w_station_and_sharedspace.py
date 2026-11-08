from io import BytesIO
import tarfile

import numpy as np

from scripts.audit_m3w_station_and_sharedspace import aschaffenburg,grid_support,read_station
from scripts.audit_m3w_tumdot_ing_support import frame_mapping
from src.data_unification.m3w_topdown_source_support import support


def test_fast_structural_support_matches_past_only_reference():
    rng = np.random.default_rng(81)
    tracks = [np.flatnonzero(rng.random(48)>.08) for _ in range(14)]
    assert grid_support([v+12 for v in tracks],(12,59))==support(tracks,48,1)
    assert grid_support([v+12 for v in tracks],(12,59),2)==support(tracks,48,2)


def test_clock_conflicts_and_duplicates_do_not_get_interpolated():
    fixture = ('0,1,2,3,4\n1,0,2023-01-01 00:00:00,?,?\n1,3,2023-01-01 00:00:01,?,?\n'
               '2,0,2023-01-01 00:00:00,?,?\n2,0,2023-01-01 00:00:00,?,?\n'
               '3,0,2023-01-01 00:00:00,?,?\n3,2,2023-01-01 00:00:02,?,?\n')
    r,keys = read_station(fixture.encode(),'2023-01-01')
    assert keys=={1,2,3} and r['verified_tracks']==1
    assert r['clock_inconsistent_tracks']==r['duplicate_tracks']==1


def test_segment_sample_count_does_not_imply_multiagent_clock(tmp_path):
    path = tmp_path/'pedestrians.tar.gz'
    with tarfile.open(path,'w:gz') as z:
        raw = b'N=20 fps=50.000000\n'+b'x y a b c\n'*20
        for label in ('start','walk'):
            member = tarfile.TarInfo(f'pedestrians/{label}/1_1.txt')
            member.size = len(raw)
            z.addfile(member,BytesIO(raw))
    a = aschaffenburg(path)
    assert a['row_ordinal_eligible_segments']=={'stride_1':2,'stride_20':0}
    assert a['segment_stems_reused']==1 and not a['shared_multiagent_windows_verified']


def test_relative_clock_accepts_recording_specific_offset_and_detects_gap():
    t = 0.14+np.array([0,1,2,4,5])/29.97
    rows,err,gaps = frame_mapping(t)
    assert list(rows.values())==[0,1,2,4,5] and err<1e-8 and gaps==1
