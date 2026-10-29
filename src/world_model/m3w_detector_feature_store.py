"""Read only verified past detector fields on admitted fitting row identities."""
import json
from pathlib import Path

import numpy as np

from scripts.fetch_m3w_european_squares import digest
from src.world_model.m3w_detector_cost import representation


class DetectorFeatureStore:
    def __init__(self,root,public):
        root,public=Path(root),Path(public)
        report=json.loads((public/'complete.json').read_text());check=json.loads((public/'verification.json').read_text())
        if (not check['raw_reparsed'] or not check['all_arrays_exact']
                or check['complete_sha256']!=digest(public/'complete.json')
                or report['query_future_labels_read'] or report['independent_roles_opened']
                or report['rows']!=149510 or report['recordings']!=74):
            raise ValueError('Full verified original fitting-only detector store required')
        self.report=report;self.complete_sha256=digest(public/'complete.json');self.arrays={}
        for key,ref in report['arrays'].items():
            path=root/ref['path']
            if digest(path)!=ref['sha256']:raise ValueError('Frozen private detector array changed:'+key)
            self.arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
        ids=self.arrays['ids']
        if np.any(np.diff(ids)<=0):raise ValueError('Sorted distinct original row IDs required')

    def read(self,ids,arm):
        ids=np.asarray(ids)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer):
            raise ValueError('Explicit integer fitting-row IDs required')
        position=np.searchsorted(self.arrays['ids'],ids)
        if np.any(position>=len(self.arrays['ids'])) or not np.array_equal(self.arrays['ids'][position],ids):
            raise ValueError('Row outside the admitted fitting detector cache')
        return representation(self.arrays['features'][position],self.arrays['recordings'][position],
                              self.arrays['agents'][position],self.arrays['frames'][position],arm)
