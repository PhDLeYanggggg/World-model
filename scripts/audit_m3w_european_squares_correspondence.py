"""Additional recording/site correspondence checks without changing source IDs."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_european_squares import RAW, digest, save
from scripts.audit_m3w_european_squares_v2 import PUBLIC


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    comp=pd.read_csv(RAW/'stats_comparative.csv',usecols=['No.','City','Country','Date','timeslot'])
    season=pd.read_csv(RAW/'stats_season.csv',usecols=['No.','City','Date','timeslot'])
    records=[r for r in a['recordings'] if 'Trajectories_comparative/' in r['source_member']]
    raw_count=Counter(r['square_id'] for r in records)
    meta_count=Counter(map(int,comp['No.']))
    differing=[]
    for sid in sorted(set(raw_count)|set(meta_count)):
        if raw_count[sid]!=meta_count[sid]:
            differing.append(dict(square_id=sid,raw_count=raw_count[sid],metadata_count=meta_count[sid],
                metadata=comp[comp['No.']==sid].to_dict('records'),
                raw_members=[r['source_member'] for r in records if r['square_id']==sid]))
    raw_date=Counter()
    unparsed=[]
    for r in records:
        match=re.match(r'(\d+)_(\d{4})-(\d{2})-(\d{2})-',Path(r['source_member']).name)
        if not match:
            unparsed.append(r['source_member'])
            continue
        sid,y,m,d=match.groups()
        raw_date[(int(sid),int(y+m+d))]+=1
    meta_date=Counter((int(r['No.']),int(r['Date'])) for _,r in comp.iterrows())
    date_diffs=[dict(square_id=k[0],date=k[1],raw_count=raw_date[k],metadata_count=meta_date[k])
                for k in sorted(set(raw_date)|set(meta_date)) if raw_date[k]!=meta_date[k]]
    season_keys=Counter((str(r['City']).casefold(),int(r['Date']),str(r['timeslot'])) for _,r in season.iterrows())
    result=dict(result_source='fresh_run',analysis_sha256=digest(PUBLIC/'analysis.json'),
        script_sha256=digest(Path(__file__)),comparative_raw_recordings=len(records),
        comparative_metadata_records=len(comp),comparative_site_count_differences=differing,
        comparative_site_date_count_differences=date_diffs,unparsed_comparative_dates=unparsed,
        season_unique_city_date_slot_keys=len(season_keys),
        season_repeated_city_date_slot_keys=[dict(city=k[0],date=k[1],timeslot=k[2],count=v) for k,v in sorted(season_keys.items()) if v>1],
        interpretation='same_total_count_does_not_establish_recording_correspondence',
        source_ids_modified=False,forecast_errors_opened=False)
    save(PUBLIC/'comparative_correspondence.json',result)
    print(json.dumps(dict(site_count_differences=len(differing),site_date_count_differences=len(date_diffs),
                         unparsed_dates=len(unparsed),season_unique_keys=len(season_keys))))


if __name__=='__main__':
    main()
