"""Structural support audit for one fixed public TUMDOT-ING file per intersection."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_m3w_station_and_sharedspace import grid_support
from scripts.audit_m3w_topdown_sources import identity

FILES = {
    'P1':('Intersection_P1_-_d1_-__15_54_47_16_01_40__-_A.csv',44641777),
    'P2':('Intersection_2-d1-_15_54_31_16_01_2.csv',67274209),
    'P3':('Intersection_3-d1-_15_54_28_16_01_0.csv',54035927),
}


def frame_mapping(timestamps):
    t = np.asarray(sorted(timestamps))
    raw = (t-t[0])*29.97
    ticks = np.rint(raw).astype(np.int64)
    err = np.abs((raw-ticks)/29.97)
    if err.max() > 2e-6 or (np.diff(ticks)<=0).any():
        raise ValueError('Cannot establish a unique 29.97 Hz relative release grid')
    return dict(zip(t,ticks,strict=True)), float(err.max()),int((np.diff(ticks)>1).sum())


def read(path,size):
    if path.stat().st_size!=size:
        raise ValueError('Local file does not match publisher listing size')
    timestamps = set()
    tracks = defaultdict(lambda:defaultdict(list))
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows = csv.reader(f)
        if next(rows)[:3]!=['timestep','track_id','category']:
            raise ValueError('Unexpected TUMDOT-ING schema')
        for row in rows:
            ts,key,kind = row[:3]
            val = float(ts)
            if not np.isfinite(val) or not key or not kind:
                raise ValueError('Malformed TUMDOT-ING metadata')
            timestamps.add(val)
            tracks[kind][key].append(val)
    ticks, err, gaps = frame_mapping(timestamps)
    unique = set().union(*(set(a) for a in tracks.values()))
    if len(unique)!=sum(map(len,tracks.values())):
        raise ValueError('TUMDOT-ING track changes class')
    ped = {k:v for k,v in tracks.items() if k.lower() in ('pedestrian','pedestrians')}
    if len(ped)!=1:
        raise ValueError(f'Unrecognized pedestrian classes: {list(tracks)}')
    typ,p = next(iter(ped.items()))
    good = [[ticks[t] for t in seq] for seq in p.values() if len(seq)==len(set(seq))]
    return dict(asset=identity(path),release_grid_hz=29.97,frame_times=len(ticks),
        maximum_grid_residual_seconds=err,gaps_in_timeline=gaps,
        observations_by_class={k:sum(map(len,v.values())) for k,v in tracks.items()},
        tracks_by_class={k:len(v) for k,v in tracks.items()},pedestrian_class=typ,
        duplicate_pedestrian_tracks=len(p)-len(good),**{
            f'stride_{stride}':grid_support(good,(0,max(ticks.values())),stride) for stride in (1,12)}),unique


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    sites,keys = {},{}
    for k,(name,n) in FILES.items():
        sites[k],keys[k] = read(a.directory/name,n)
    counts = Counter(x for seq in keys.values() for x in seq)
    result = dict(scope='structural_only',sites=sites,campaigns=1,locations=3,
                  cross_sample_shared_numeric_or_text_ids=sum(n>1 for n in counts.values()),
                  file_rule='earliest_day_one_listed_record_per_intersection',
                  shared_city=True,scored_forecasts=False,assigned_predictive_roles=False,
                  dronecrowd_confirmation_access=False,steps=[8,12])
    content = json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if a.output.read_text()!=content:
            raise ValueError('Structural replay differs from recorded sample')
        print('Exact structural replay matched')
    else:
        tmp = a.output.with_suffix('.pending')
        tmp.write_text(content)
        tmp.replace(a.output)
        print('TUMDOT-ING structural readout saved')


if __name__=='__main__':
    main()
