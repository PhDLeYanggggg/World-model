"""Read publisher timing and track IDs only; no predictive outcomes or role assignment."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import ssl
import sys
from tarfile import open as open_tar
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_m3w_topdown_sources import identity,totals
from src.data_unification.m3w_public_zip_range import RangeArchive

URL = 'https://ndownloader.figshare.com/files/49279738'
LENGTH = 3741410989
MD5 = 'b6e4d096c18afc235d46d5e034685263'
PATTERN = re.compile(r'Data-2/((?:HB|PB)-CAM\d+)/\1_(\d{4}-\d{2}-\d{2})_trans\.csv')


def grid_support(tracks, extent, stride=1):
    start,stop = extent
    n = max(0,stop-start+1-19*stride)
    observed = np.zeros(n,dtype=np.int64)
    complete = np.zeros_like(observed)
    stats = dict(past_eligible=0,endpoint=0,complete12=0,partial12=0,no_future=0,
                 eligible_tracks=0,complete_tracks=0)
    for track in tracks:
        if not n or len(track)<8:
            continue
        t = np.asarray(sorted(track),dtype=np.int64)
        if t[0]<start or t[-1]>stop or (np.diff(t)<=0).any():
            raise ValueError('Expected unique times in the reported extent')
        at = t[t+19*stride<=stop]
        at = at[np.isin(at[:,None]+stride*np.arange(1,8),t).all(1)]
        if not len(at):
            continue
        mask = np.isin(at[:,None]+stride*np.arange(8,20),t)
        full = mask.all(1)
        observed[at-start] += 1
        complete[at[full]-start] += 1
        stats['past_eligible'] += len(at)
        stats['endpoint'] += int(mask[:,-1].sum())
        stats['complete12'] += int(full.sum())
        stats['partial12'] += int((mask.any(1)&~full).sum())
        stats['no_future'] += int((~mask.any(1)).sum())
        stats['eligible_tracks'] += 1
        stats['complete_tracks'] += int(full.any())
    pos = np.flatnonzero(complete>=2)
    next_ = count = 0
    for i in pos:
        if i>=next_:
            count += 1
            next_ = i+19*stride+1
    return {**stats,'queries':int((observed>0).sum()),'multiagent_queries':int((observed>=2).sum()),
            'multiagent_complete_queries':len(pos),'nonoverlapping_multiagent_complete':count}


def read_station(data,day):
    rows = csv.reader(StringIO(data.decode('utf-8-sig')))
    if next(rows)!=['0','1','2','3','4']:
        raise ValueError('Station CSV has unrecognized positional headers')
    tracks = defaultdict(list)
    for cols in rows:
        if len(cols)!=5:
            raise ValueError('Malformed station CSV row')
        ident,elapsed,date = cols[:3]
        ts = datetime.fromisoformat(date)
        if ts.tzinfo is not None or ts.microsecond or ts.date().isoformat()!=day:
            raise ValueError('Station timestamps are not date-local integer seconds')
        stamp = ts.hour*3600+ts.minute*60+ts.second
        rel = float(elapsed)
        if not np.isfinite(rel):
            raise ValueError('Nonfinite relative time')
        tracks[int(ident)].append((stamp,rel))
    if not tracks:
        raise ValueError('Unexpected empty station member')
    verified = []
    duplicate = clock = 0
    intervals = Counter()
    for track in tracks.values():
        series = sorted(track)
        stamps = [t for t,_ in series]
        elapsed = [r for _,r in series]
        dt = np.diff(stamps)
        duplicated = bool((dt==0).any())
        inconsistent = bool((np.abs(np.diff(elapsed)-dt)>1e-6).any())
        duplicate += duplicated
        clock += inconsistent
        intervals.update(map(int,dt))
        if not duplicated and not inconsistent:
            verified.append(stamps)
    extent = (min(t for a in tracks.values() for t,_ in a),max(t for a in tracks.values() for t,_ in a))
    return dict(observations=sum(map(len,tracks.values())),tracks=len(tracks),verified_tracks=len(verified),
                duplicate_tracks=duplicate,clock_inconsistent_tracks=clock,release_interval_counts={
                    'same_second':intervals[0],'one_second':intervals[1],
                    'above_one_second':sum(n for k,n in intervals.items() if k>1),'maximum':max(intervals,default=None)},
                **grid_support(verified,extent)), set(tracks)


def station(cafile):
    context = ssl.create_default_context(cafile=cafile)
    records = {}
    ids = defaultdict(list)
    with RangeArchive(URL,LENGTH,context=context,byte_limit=96<<20) as f, ZipFile(f) as z:
        directory = defaultdict(dict)
        for i in z.infolist():
            hit = PATTERN.fullmatch(i.filename)
            if not hit:
                continue
            camera,date = hit.groups()
            if date in directory[camera]:
                raise ValueError('Duplicate camera-day member in archive')
            directory[camera][date] = i
        if len(directory)!=11 or sum(map(len,directory.values()))!=3184:
            raise ValueError('Expected the publisher v1 directory')
        common = sorted(set.intersection(*(set(x) for x in directory.values())))
        days = [common[0],common[len(common)//2],common[-1]]
        for camera in sorted(directory):
            print(f'Reading structural sample: {camera}',flush=True)
            for day in days:
                i = directory[camera][day]
                if i.file_size>64<<20:
                    raise ValueError('Selected sample exceeds the per-member bound')
                raw = z.read(i)
                stats,keys = read_station(raw,day)
                ids[camera[:2],day].append(keys)
                records[i.filename] = {**stats,'sha256':sha256(raw).hexdigest(),'crc32':f'{i.CRC:08x}',
                                       'compressed_bytes':i.compress_size}
        source = dict(bytes=LENGTH,publisher_md5_not_recomputed=MD5,range_bytes_fetched=f.total,
                      storage_version=f.version,full_archive_downloaded=False)
    overlaps = {}
    for (site,day),views in ids.items():
        all_ids = Counter(t for ids_ in views for t in ids_)
        overlaps[f'{site}/{day}'] = sum(n>1 for n in all_ids.values())
    keys = ('tracks','observations','verified_tracks','duplicate_tracks','clock_inconsistent_tracks','past_eligible',
            'endpoint','complete12','partial12','no_future','eligible_tracks','complete_tracks','queries',
            'multiagent_queries','multiagent_complete_queries','nonoverlapping_multiagent_complete')
    return dict(source=source,files_by_camera={k:len(v) for k,v in directory.items()},common_dates=len(common),
        sampled_days=days,roster_selection='first_median_last_date_shared_by_all_cameras',members=records,
        same_numeric_id_across_cameras=overlaps,acquisition_stations=2,aggregate_by_station={site:totals(
            {k:r[k] for k in keys} for name,r in records.items() if name.startswith(f'Data-2/{site}-')) for site in ('HB','PB')})


def hannover(path):
    records = {}
    with ZipFile(path) as z:
        for clip in ('D1','D2','D3'):
            raw = z.read(f'trajectory_data/{clip}_trajectory.csv')
            rows = csv.reader(StringIO(raw.decode('utf-8-sig')))
            if next(rows)!=['time_step','object_id','x_coordinate','y_coordinate','user_type']:
                raise ValueError('Unexpected trajectory CSV schema')
            tracks = defaultdict(lambda:defaultdict(list))
            low,high = float('inf'),float('-inf')
            for step,ident,_,_,kind in rows:
                if kind not in ('1','2','3'):
                    raise ValueError('Unknown Hannover road user type')
                frame = int(step)
                low,high = min(low,frame),max(high,frame)
                tracks[kind][int(ident)].append(frame)
            if len(set().union(*(set(v) for v in tracks.values())))!=sum(map(len,tracks.values())):
                raise ValueError('Object has multiple road user types')
            ped = tracks['1']
            verified = [v for v in ped.values() if len(v)==len(set(v))]
            records[clip] = dict(sha256=sha256(raw).hexdigest(),tracks={k:len(v) for k,v in tracks.items()},
                observations={k:sum(map(len,v.values())) for k,v in tracks.items()},
                duplicate_ped_tracks=len(ped)-len(verified),**grid_support(verified,(low,high)))
    return dict(asset=identity(path),clips=records,annotation_tick_seconds=.5,provisional_acquisition_groups=1)


def aschaffenburg(path):
    layout = Counter()
    actions = Counter()
    fps = Counter()
    lengths = []
    base_ids = Counter()
    stems = Counter()
    seen = set()
    with open_tar(path,'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            match = re.fullmatch(r'pedestrians/(walk|stand|start|stop)/(\d+)_(\d+)\.txt',member.name)
            if not member.isfile() or match is None or member.size>2<<20 or member.name in seen:
                raise ValueError('Unexpected member in publisher pedestrian archive')
            seen.add(member.name)
            action,ident,part = match.groups()
            with archive.extractfile(member) as f:
                header = f.readline().decode('ascii').strip()
                m = re.fullmatch(r'N=(\d+) fps=(\d+(?:\.\d+)?)',header)
                if m is None:
                    raise ValueError('Unknown Aschaffenburg publisher header')
                n,rate = int(m[1]),m[2]
                widths = Counter(len(row.split()) for row in f)
                if sum(widths.values())!=n:
                    raise ValueError('Aschaffenburg row count differs from the header')
            layout.update(widths)
            actions[action] += 1
            fps[rate] += 1
            lengths.append(n)
            base_ids[ident] += 1
            stems[ident,part] += 1
    if not lengths:
        raise ValueError('No trajectory segments in pedestrian archive')
    return dict(asset=identity(path),segments=len(lengths),rows=sum(lengths),
                fps_header_counts=dict(fps),payload_column_counts=dict(layout),actions=dict(actions),
                distinct_filename_prefixes=len(base_ids),prefixes_reused=sum(n>1 for n in base_ids.values()),
                segment_stems_reused=sum(n>1 for n in stems.values()),length_min=min(lengths),
                length_median=float(np.median(lengths)),length_max=max(lengths),
                row_ordinal_eligible_segments={f'stride_{s}':sum(n>=19*s+1 for n in lengths) for s in (1,20)},
                published_common_clock_verified=False,shared_multiagent_windows_verified=False,
                acquisition_groups=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--hannover-archive',type=Path)
    sources.add_argument('--aschaffenburg-only',type=Path)
    p.add_argument('--cafile',default=None)
    a = p.parse_args()
    result = dict(scope='structural_only',scored_forecasts=False,assigned_predictive_roles=False,
        dronecrowd_confirmation_access=False,steps=[8,12])
    if a.aschaffenburg_only:
        result['aschaffenburg'] = aschaffenburg(a.aschaffenburg_only)
    else:
        result.update(hannover=hannover(a.hannover_archive),station=station(a.cafile))
    payload = json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if a.output.read_text()!=payload:
            raise ValueError('Audit does not match frozen output')
        print('Exact structural replay matched',flush=True)
    else:
        pending = a.output.with_suffix('.pending')
        pending.write_text(payload)
        pending.replace(a.output)
        print('Structural evidence written',flush=True)


if __name__=='__main__':
    main()
