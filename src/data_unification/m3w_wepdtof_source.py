"""Parse publisher frame and identity support without borrowing future positions."""
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np

from src.evaluation.m3w_opendd_location import cohorts


def source(clip):
    return clip[:-2] if clip in ('large_office_2', 'jewelry_store_2') else clip


def frame_number(name, clip):
    prefix = clip + '_'
    stem = PurePosixPath(name).stem
    if not stem.startswith(prefix) or not stem[len(prefix):].isdigit():
        raise ValueError(f'Unexpected frame name: {name}')
    return int(stem[len(prefix):])


def catalog(members):
    clips = defaultdict(dict)
    for n in members:
        p = PurePosixPath(n)
        if len(p.parts) != 4 or p.parts[:2] != ('WEPDTOF', 'frames') or p.suffix.lower() != '.jpg':
            continue
        clip = p.parts[2]
        tick = frame_number(p.name, clip)
        if tick in clips[clip]:
            raise ValueError('Duplicate image tick')
        clips[clip][tick] = n
    if not clips:
        raise ValueError('No publisher frames')
    return dict(clips)


def annotation_table(doc, clip, frames, geometry=False):
    ids, ticks = {}, set()
    for row in doc['images']:
        file = row['file_name']
        t = frame_number(file, clip)
        if (row['id'] in ids or t in ticks or t not in frames
                or PurePosixPath(frames[t]).name != file):
            raise ValueError('Image ID/tick does not match archive')
        ids[row['id']] = t
        ticks.add(t)
    if not ticks or [(c['id'],c['name']) for c in doc['categories']] != [(1,'person')]:
        raise ValueError('Unexpected annotation population')
    keys = Counter((int(r['person_id']),ids[r['image_id']]) for r in doc['annotations'])
    ambiguous = {k for k,n in keys.items() if n > 1}
    people = defaultdict(dict)
    for r in doc['annotations']:
        t, person = ids[r['image_id']], int(r['person_id'])
        if r['category_id'] != 1:
            raise ValueError('Unexpected identity label')
        if (person,t) in ambiguous:
            continue
        if geometry:
            b = np.asarray(r['bbox'], dtype=float)
            if b.shape != (5,) or not np.isfinite(b).all() or min(b[2:4]) <= 0.:
                raise ValueError('Expected finite rotated box')
            people[person][t] = b
        else:
            people[person][t] = True
    return dict(people), dict(raw_rows=len(doc['annotations']), raw_track_ids=len({k[0] for k in keys}),
                              ambiguous_identity_ticks=len(ambiguous), excluded_rows=sum(keys[k] for k in ambiguous),
                              annotation_image_entries=len(ticks), unlisted_images=len(frames)-len(ticks))


def support(people, frames, stride):
    ticks = np.array(sorted(frames),dtype=int)
    spans = np.split(ticks,np.flatnonzero(np.diff(ticks)>1)+1)
    shifts = stride*np.arange(8,20)
    n = []
    for t in spans:
        mask = np.zeros((len(people), len(t)), dtype=bool)
        for j, times in enumerate(people.values()):
            mask[j, [s-t[0] for s in times if t[0]<=s<=t[-1]]] = True
        n.extend(mask[rows[:,None],start+shifts].sum(1) for start, rows in cohorts(mask, stride))
    complete = sum(int((a == 12).sum()) for a in n)
    absent = sum(int((a == 0).sum()) for a in n)
    total = sum(len(a) for a in n)
    return dict(queries=len(n), targets=total, complete=complete, absent=absent,
                partial=total-complete-absent, queries_with_two_complete=sum(int((a == 12).sum()>=2) for a in n))


def pilot_starts(frames, stride, count):
    ticks = sorted(frames)
    base = np.arange(ticks[0],ticks[-1]-19*stride+1,19*stride+1)
    if len(base) < count or ticks != list(range(ticks[0],ticks[-1]+1)):
        raise ValueError('Pilot needs sufficient continuous frame support')
    return base[np.linspace(0,len(base)-1,count,dtype=int)].tolist()
