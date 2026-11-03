"""Sparse official MOT intake and diagnostic native-step windows, not admission."""
from collections import OrderedDict
import csv
import hashlib
import io
from pathlib import PurePosixPath
import re
from urllib.parse import urlparse

import numpy as np
import requests

from src.evaluation.m3w_dronecrowd_image_intake import content_range
from src.evaluation.m3w_dronecrowd_windows import (
    Recording, WindowSpec, compare_inputs, past_inputs, target_labels,
)


def member_path(name):
    p = PurePosixPath(name)
    if (not name or '\\' in name or p.is_absolute()
            or any(v in ('..','') for v in name.rstrip('/').split('/'))):
        raise ValueError('Unsafe archive member path')
    return p


def clip_key(name, kind):
    p = member_path(name)
    if p.parts.count(kind) != 1:
        return None
    rest = p.parts[p.parts.index(kind)+1:]
    if kind == 'annotations' and len(rest) == 2 and p.suffix.lower() == '.csv':
        return '/'.join((rest[0],p.stem))
    if kind == 'frames' and len(rest) == 3 and p.suffix.lower() in ('.jpg','.jpeg','.png'):
        if not re.fullmatch(r'[1-9][0-9]*',p.stem):
            raise ValueError('Published image indices must be positive integer names')
        return '/'.join(rest[:2])
    return None


class PublicRangeZip(io.RawIOBase):
    """A bounded, immutable-revision HTTP view; full-download fallbacks are refused."""
    def __init__(self,url,size,*,maximum_bytes,block_size=262144,session=None):
        super().__init__()
        if (urlparse(url).scheme != 'https' or urlparse(url).hostname != 'huggingface.co'
                or not re.fullmatch(r'/datasets/fyw1999/MovingDroneCrowd/resolve/[0-9a-f]{40}/MovingDroneCrowd%2B%2B.zip',urlparse(url).path)):
            raise ValueError('Only the pinned official public dataset archive is allowed')
        if type(size) is not int or size <= 0 or maximum_bytes <= 0:
            raise ValueError('Positive fixed archive size and transfer cap required')
        self.url,self.size,self.maximum_bytes = url,size,maximum_bytes
        self.block_size,self.position,self.transferred_bytes,self.range_requests = block_size,0,0,0
        self.cache = OrderedDict()
        self.session = session or requests.Session()

    def fetch(self,start,end):
        length = end-start+1
        if not 0 <= start <= end < self.size or self.transferred_bytes+length > self.maximum_bytes:
            raise ValueError('Out-of-range or over-budget archive request')
        with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}',
                'Accept-Encoding':'identity'},stream=True,timeout=(10,60)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError('Server ignored byte range; refusing full download')
            host = urlparse(response.url).hostname or ''
            if (host != 'huggingface.co' and not host.endswith('.hf.co')
                    and not host.endswith('.xethub.hf.co')):
                raise ValueError('Unexpected archive redirect host')
            if response.headers.get('Content-Encoding','identity') != 'identity':
                raise ValueError('Byte ranges must use identity encoding')
            if content_range(response.headers.get('Content-Range')) != (start,end,self.size):
                raise ValueError('Archive range or total size changed')
            data = response.raw.read(length+1)
            if len(data) != length:
                raise ValueError('Truncated or oversized archive range')
        self.transferred_bytes += length
        self.range_requests += 1
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self,offset,whence=0):
        base = {0:0,1:self.position,2:self.size}.get(whence)
        if base is None or not 0 <= base+offset <= self.size:
            raise ValueError('Invalid seek')
        self.position = base+offset
        return self.position

    def read(self,size=-1):
        end = self.size if size < 0 else min(self.size,self.position+size)
        if end-self.position > 32*1024**2:
            raise ValueError('Refusing archive-wide or oversized member read')
        pieces = []
        while self.position < end:
            block = self.position//self.block_size
            if block not in self.cache:
                start = block*self.block_size
                self.cache[block] = self.fetch(start,min(start+self.block_size,self.size)-1)
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
            raw = self.cache[block]
            self.cache.move_to_end(block)
            offset = self.position%self.block_size
            count = min(end-self.position,len(raw)-offset)
            pieces.append(raw[offset:offset+count])
            self.position += count
        return b''.join(pieces)

    def close(self):
        if hasattr(self,'session'):
            self.session.close()
        super().close()


def mot_record(data,key,image_frames,*,maximum_cells=20000000):
    """Strict released rows; all future support remains outside input creation."""
    values = []
    for number,row in enumerate(csv.reader(io.StringIO(data.decode('utf-8-sig'))),1):
        if not row or all(not v.strip() for v in row):
            continue
        if len(row) != 10:
            raise ValueError(f'MOT row{number} does not have ten columns')
        a = np.asarray([float(v) for v in row],np.float64)
        if (not np.isfinite(a).all() or np.any(a[:2] != np.floor(a[:2]))
                or np.any(a[:2] < 0) or np.any(a[4:6] <= 0) or np.any(a[:2] > 2**53)):
            raise ValueError(f'Invalid MOT geometry or identity at row{number}')
        values.append(a)
    if not values:
        raise ValueError('Empty annotation file')
    rows = np.stack(values)
    if len(np.unique(rows[:,:2],axis=0)) != len(rows):
        raise ValueError('Repeated frame/agent identity; do not silently deduplicate')
    frames = np.asarray(sorted(image_frames),np.int64)
    if not np.array_equal(frames,np.arange(1,len(frames)+1)):
        raise ValueError('Released image sequence has a gap or invalid origin')
    agents = np.unique(rows[:,1]).astype(np.int64)
    if len(agents)*len(frames) > maximum_cells:
        raise ValueError('Dense recording exceeds diagnostic memory budget')
    f = rows[:,0].astype(np.int64)
    if f.max() >= len(frames):
        raise ValueError('Annotation frame does not map to image frame+1')
    positions = np.zeros((len(agents),len(frames),2),np.float64)
    valid = np.zeros(positions.shape[:2],bool)
    at = np.searchsorted(agents,rows[:,1].astype(np.int64))
    positions[at,f] = rows[:,2:4]+rows[:,4:6]/2
    valid[at,f] = True
    # The legacy internal hash field name is not source geometry or an input.
    # No image dimensions were acquired; the bounds mask is unknown, not gold.
    record = Recording(key,hashlib.sha256(data).hexdigest(),agents,positions,valid,np.zeros_like(valid))
    return record


def native_inputs(record,query,spec):
    value = past_inputs(record,query,spec)
    value.pop('in_bounds_mask')
    value.update(velocity_unit='image_pixel_per_native_annotation_step',
        frame_unit='native_annotation_step_not_verified_raw_video_frame',
        geometry='released_MOT_head_box_center',data_role='diagnostic_only')
    return value


def native_labels(record,inputs,spec):
    # Rebuild the unmodified common envelope for its source/prefix check.
    if not compare_inputs(inputs,native_inputs(record,inputs['query_frame'],spec)):
        raise ValueError('Changed diagnostic inputs')
    return target_labels(record,past_inputs(record,inputs['query_frame'],spec),spec)


def endpoint_support(record,history,offset):
    """Require contiguous history; endpoint labels need not fill intervening gaps."""
    total = continuous = 0
    for mask in record.valid:
        visible = np.flatnonzero(mask)
        if not len(visible):
            continue
        runs = np.split(visible,np.flatnonzero(np.diff(visible)!=1)+1)
        for run in runs:
            candidates = run[history-1:]
            eligible = candidates[candidates+offset < len(mask)]
            total += int(mask[eligible+offset].sum())
            continuous += max(0,len(run)-(history-1+offset))
    return dict(endpoint_label_windows=total,complete_interval_windows=continuous)
