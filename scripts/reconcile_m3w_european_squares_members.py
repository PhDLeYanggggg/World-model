"""Compare publisher recording metadata with actual archive members; no repair."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_european_squares import RAW, PUBLIC as SOURCE_PUBLIC, digest, save
from scripts.audit_m3w_european_squares_v2 import PUBLIC


def main():
    source=json.loads((SOURCE_PUBLIC/'trajectory_manifest.json').read_text())
    archive=ROOT/source['private_file']['path']
    if digest(archive)!=source['private_file']['sha256']:
        raise ValueError('Archive differs')
    schemas, raw_season, by_square = defaultdict(list), defaultdict(list), Counter()
    inventory=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=info.filename
            if name.startswith('__MACOSX/') or info.is_dir():
                continue
            inventory.append(dict(path=name, bytes=info.file_size))
            if 'Trajectories_raw/' not in name or not name.endswith('.csv'):
                continue
            with z.open(name) as f:
                header=f.readline().decode('utf-8-sig').strip()
            schemas[header].append(name)
            square=int(Path(name).name.split('_')[0])
            by_square[square]+=1
            if 'Trajectories_season/' in name:
                match=re.match(r'(\d+)_(\d{8})_(morning|noon|evening)_([^_]+)_',Path(name).name)
                if not match:
                    raise ValueError('Unrecognized season filename '+name)
                sid,date,slot,city=match.groups()
                raw_season[(city.casefold(),int(date),slot)].append(dict(path=name,square_id=int(sid)))
    stats=pd.read_csv(RAW/'stats_season.csv',usecols=['No.','City','Date','timeslot'])
    missing, conflicts, metadata_keys, matches = [], [], set(), 0
    for _,r in stats.iterrows():
        key=(str(r['City']).casefold(),int(r['Date']),str(r['timeslot']))
        metadata_keys.add(key)
        candidates=raw_season.get(key,[])
        rec=dict(city=r['City'],date=int(r['Date']),timeslot=r['timeslot'],metadata_square_id=int(r['No.']))
        if not candidates:
            missing.append(rec)
        else:
            matches+=1
            if len(candidates)!=1 or candidates[0]['square_id']!=int(r['No.']):
                conflicts.append(dict(**rec,raw_members=candidates))
    result=dict(result_source='fresh_run',archive_sha256=source['private_file']['sha256'],
                script_sha256=digest(Path(__file__)),
                schema_counts={k:len(v) for k,v in schemas.items()},
                raw_recordings=sum(map(len,schemas.values())),
                raw_recordings_by_nominal_square={str(k):v for k,v in sorted(by_square.items())},
                season_metadata_records=len(stats),season_raw_members=sum(map(len,raw_season.values())),
                season_metadata_records_with_name_date_slot_match=matches,
                season_metadata_missing_raw=missing,season_identity_disagreements=conflicts,
                season_raw_without_stats_match=[v for k,v in raw_season.items() if k not in metadata_keys],
                total_nonmac_file_bytes=sum(r['bytes'] for r in inventory),
                raw_uncompressed_bytes=sum(r['bytes'] for r in inventory if 'Trajectories_raw/' in r['path'] and r['path'].endswith('.csv')),
                metadata_repaired=False,processed_substituted=False,forecast_errors_opened=False)
    save(PUBLIC/'archive_reconciliation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['raw_recordings','season_metadata_records',
                    'season_raw_members','season_metadata_records_with_name_date_slot_match']}))
    print('unmatched',len(missing),'identity_disagreements',len(conflicts))


if __name__=='__main__':
    main()
