"""Reference-image pixel boxes to resized video pixels; no metric conversion."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SDDImageCoordinates:
    annotation_width: int
    annotation_height: int
    video_width: int
    video_height: int

    def __post_init__(self):
        sizes = (self.annotation_width, self.annotation_height, self.video_width, self.video_height)
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x <= 0 for x in sizes):
            raise ValueError('Positive integer reference/video dimensions required')

    @property
    def scale_xy(self):
        return np.array([self.video_width/self.annotation_width,
                         self.video_height/self.annotation_height], dtype=float)

    def past_boxes(self, boxes, *, frame_ids, query_frame):
        boxes, frames = np.asarray(boxes, dtype=float), np.asarray(frame_ids, dtype=float)
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or frames.shape != (len(boxes),)
                or not np.isfinite(boxes).all() or not np.isfinite(frames).all()
                or not isinstance(query_frame, (int, np.integer)) or isinstance(query_frame, bool)
                or query_frame < 0 or np.any(frames < 0) or np.any(frames != np.rint(frames))
                or np.any(frames > query_frame) or np.any(boxes[:, 2:] < boxes[:, :2])):
            raise ValueError('Only finite, ordered past/current boxes at integral frames are allowed')
        # Continuous box edges scale without changing the annotation-space state.
        return boxes*np.tile(self.scale_xy, 2)

    def video_boxes_to_annotation(self, boxes):
        boxes = np.asarray(boxes, dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError('Finite Nx4 video boxes required')
        return boxes/np.tile(self.scale_xy, 2)

    def metadata(self):
        return dict(annotation_size=[self.annotation_width, self.annotation_height],
            video_size=[self.video_width, self.video_height], scale_xy=self.scale_xy.tolist(),
            convention='continuous_xyxy_box_edges_reference_to_video',
            state_coordinate_unit='annotation_pixel_unchanged', crop_coordinate_unit='video_pixel',
            homography_or_metric_calibration=False, future_inputs_allowed=False,
            semantic_alignment_requires_source_verification=True)
