"""Explicit image axes and past frame-index requests for local media inspection."""
from __future__ import annotations

import numpy as np


def native_to_image_xy(points, homography, *, projected_axes):
    """Supplied-H inversion is not physical calibration or temporal alignment."""
    xy,h=np.asarray(points,float),np.asarray(homography,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or h.shape!=(3,3) or not np.isfinite(xy).all() or not np.isfinite(h).all():
        raise ValueError('Finite native positions and 3x3 matrix required')
    q=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(h).T
    if np.any(np.abs(q[:,2])<=1e-12):
        raise ValueError('Undefined projective point')
    projected=q[:,:2]/q[:,2:]
    if projected_axes=='row_col':
        return projected[:,::-1]
    if projected_axes=='xy':
        return projected
    raise ValueError('An explicit source axis convention is required')


def past_frame_indices(current_frame, history_frames):
    """Validate the approved eight observed indices, never infer seconds or offsets."""
    t=np.asarray(history_frames)
    if (t.shape!=(8,) or not np.isfinite(t).all() or not np.all(t==np.floor(t))
            or not np.isfinite(current_frame) or current_frame!=int(current_frame)
            or np.any(t<0) or not np.all(np.diff(t)>0) or t[-1]!=current_frame
            or np.any(t>current_frame)):
        raise ValueError('Exactly eight ordered nonnegative past/current frame indices required')
    return t.astype(np.int64)


def requested_crop_box(image_xy, width, height):
    """Fixed inspection crop around an annotation point, not a detected body box."""
    xy=np.asarray(image_xy,float)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or width<=0 or height<=0:
        raise ValueError('Finite image point and positive dimensions required')
    x,y=(int(value) for value in np.rint(xy))
    wanted=(x-48,y-80,x+48,y+16)
    clipped=(max(0,wanted[0]),max(0,wanted[1]),min(width,wanted[2]),min(height,wanted[3]))
    if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:
        return None
    return {'box':clipped,'full_support':clipped==wanted,'point_inside':bool(0<=x<width and 0<=y<height)}
