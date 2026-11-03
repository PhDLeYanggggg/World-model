"""Reproducible view preparation and cross-source exclusion evidence."""
import csv
import hashlib
import io
import itertools
import re

import cv2
import numpy as np

from src.evaluation.m3w_dronecrowd_image_geometry import candidate_components


def selected_frames(frames):
    values=sorted(set(frames))
    if len(values)<3 or values[0]<1:
        raise ValueError('Three distinct audit views must be available')
    selected=[values[0],values[(len(values)-1)//2],values[-1]]
    if len(set(selected))!=3:raise ValueError('Do not count duplicate views as distinct')
    return selected


def moving_node(key):
    if not re.fullmatch(r'scene_[1-9][0-9]*/[1-9][0-9]*',key):
        raise ValueError('Invalid source clip namespace')
    return 'moving_'+key.replace('/','__')


def frame_boxes(body,frame_zero_based):
    boxes=[]
    for row in csv.reader(io.StringIO(body.decode('utf-8-sig'))):
        if not row or all(not v.strip() for v in row):continue
        if len(row)!=10:raise ValueError('Changed MOT source format')
        values=np.asarray([float(v) for v in row],float)
        if not np.isfinite(values).all() or values[0]!=int(values[0]) or np.any(values[4:6]<=0):
            raise ValueError('Changed MOT source geometry')
        if int(values[0])==frame_zero_based:
            x,y,w,h=values[2:6];boxes.append([x,y,x+w,y+h])
    return np.asarray(boxes,float).reshape(-1,4)


def prepare_raster(rgb,boxes,config):
    rgb,boxes=np.asarray(rgb),np.asarray(boxes)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or boxes.shape!=(len(boxes),4):
        raise ValueError('Decoded RGB pixels and current-image boxes required')
    width,height=config['resize']
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
    gray=cv2.createCLAHE(clipLimit=config['clahe_clip_limit'],tileGridSize=(8,8)).apply(gray)
    sx,sy=width/rgb.shape[1],height/rgb.shape[0]
    mask=np.full((height,width),255,np.uint8)
    for x0,y0,x1,y1 in boxes:
        padding=max(x1-x0,y1-y0)*.75+6
        a,c=np.clip(np.array([x0-padding,x1+padding])*sx,0,width).astype(int)
        b,d=np.clip(np.array([y0-padding,y1+padding])*sy,0,height).astype(int)
        mask[b:d,a:c]=0
    return gray,mask


def features_from_raster(gray,mask,config):
    shape=(config['resize'][1],config['resize'][0])
    if gray.shape!=shape or mask.shape!=shape or gray.dtype!=np.uint8 or mask.dtype!=np.uint8:
        raise ValueError('Lossless prepared raster/mask dimensions changed')
    points,descriptors=cv2.SIFT_create(nfeatures=config['max_features']).detectAndCompute(gray,mask)
    xy=np.asarray([p.pt for p in points],dtype=np.float32).reshape(-1,2)
    if descriptors is not None:
        descriptors=np.sqrt(descriptors/np.maximum(descriptors.sum(axis=1,keepdims=True),1e-12))
    return dict(xy=xy,descriptors=descriptors,background_fraction=float((mask>0).mean()))


def exact_features(left,right):
    if left['background_fraction']!=right['background_fraction'] or not np.array_equal(left['xy'],right['xy']):
        raise ValueError('Prepared-raster extraction changed original features')
    a,b=left['descriptors'],right['descriptors']
    if (a is None)!=(b is None) or a is not None and not np.array_equal(a,b):
        raise ValueError('Prepared-raster descriptor mismatch')


def raster_hash(gray,mask):
    return hashlib.sha256(gray.tobytes()+mask.tobytes()).hexdigest()


def pair_roster(moving,protected,seed):
    m,p=sorted(moving),sorted(protected)
    if len(set(m+p))!=len(m)+len(p):raise ValueError('Source namespaces overlap')
    within=list(itertools.combinations(m,2));cross=list(itertools.product(m,p))
    rng=np.random.default_rng(seed)
    within=[within[i] for i in rng.permutation(len(within))]
    cross=[cross[i] for i in rng.permutation(len(cross))]
    result=[]
    for a,b in itertools.zip_longest(within,cross):
        if a is not None:result.append(dict(kind='within_moving',left=a[0],right=a[1]))
        if b is not None:result.append(dict(kind='cross_source',left=b[0],right=b[1]))
    return result


def conservative_exclusions(nodes,pairs,prior_edges):
    links=[(v['left'],v['right']) for v in pairs if v['status']!='unsupported']
    groups=candidate_components(sorted(nodes),links+list(prior_edges))
    linked=sorted(node for group in groups if any(v.startswith('drone_') for v in group)
                  for node in group if node.startswith('moving_'))
    return dict(conservative_exclusion_components=groups,moving_nodes_linked_to_protected=linked,
                negative_matching_certifies_independence=False,scientific_roles_assigned=False)
