"""Past-only crop and frozen-feature diagnostics; no future labels or filtering."""
import numpy as np


def verify_history_keys(query_keys, crop_keys, image_rows, stride=12):
    query_keys, crop_keys, image_rows = map(np.asarray, (query_keys, crop_keys, image_rows))
    if (query_keys.shape != (len(image_rows), 2) or image_rows.shape != (len(query_keys), 8)
            or crop_keys.ndim != 2 or crop_keys.shape[1] != 2
            or not np.issubdtype(image_rows.dtype, np.integer)
            or np.any(image_rows < 0) or np.any(image_rows >= len(crop_keys))):
        raise ValueError('Aligned query and past-crop keys required')
    selected = crop_keys[image_rows]
    expected = query_keys[:, :1]-np.arange(7, -1, -1)*stride
    if not np.array_equal(selected[..., 0], expected):
        raise ValueError('Repeated, future or incorrectly strided image frame')
    if not np.array_equal(selected[..., 1], np.broadcast_to(query_keys[:, 1:2], expected.shape)):
        raise ValueError('Image belongs to another agent')
    return selected


def box_masks(image_boxes, size=32, crop=96):
    boxes = np.asarray(image_boxes, dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] < boxes[:, :2])):
        raise ValueError('Finite ordered image-coordinate boxes required')
    center = (boxes[:, :2]+boxes[:, 2:])/2
    origin = np.rint(center)-crop//2
    coordinate = (np.arange(size)+.5)*(crop/size)
    x = origin[:, 0, None, None]+coordinate[None, None, :]
    y = origin[:, 1, None, None]+coordinate[None, :, None]
    return ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) &
            (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))


def temporal_diagnostics(rgb, coverage, boxes, embedding):
    rgb, coverage, boxes, embedding = map(np.asarray, (rgb, coverage, boxes, embedding))
    n = len(rgb)
    if (rgb.shape != (n, 8, 3, 32, 32) or rgb.dtype != np.uint8
            or coverage.shape != (n, 8, 32, 32) or boxes.shape != (n, 8, 4)
            or embedding.shape != (n, 8, 512) or np.any(coverage > 9)
            or not np.isfinite(embedding).all()):
        raise ValueError('Aligned finite past image, coverage, box and embedding arrays required')
    supported = coverage > 0
    common = supported[:, 1:] & supported[:, :-1]
    mask = box_masks(boxes.reshape(-1, 4)).reshape(n, 8, 32, 32)
    inside = common & mask[:, 1:] & mask[:, :-1]
    outside = common & ~mask[:, 1:] & ~mask[:, :-1]
    delta = np.abs(rgb[:, 1:].astype(float)-rgb[:, :-1].astype(float)).mean(2)
    def regional_mean(region):
        count = region.sum((1, 2, 3))
        return np.divide((delta*region).sum((1, 2, 3)), count,
                         out=np.full(n, np.nan), where=count > 0), count
    total, count = regional_mean(common)
    inner, inner_count = regional_mean(inside)
    outer, outer_count = regional_mean(outside)
    valid = supported.any((2, 3))
    length = valid.sum(1)
    mean = (embedding*valid[..., None]).sum(1)/np.maximum(length[:, None], 1)
    dev = (embedding-mean[:, None])*valid[..., None]
    temporal_energy = (dev.astype(float)**2).sum((1, 2))
    energy = ((embedding.astype(float)**2)*valid[..., None]).sum((1, 2))
    fraction = np.divide(temporal_energy, energy, out=np.zeros(n), where=energy > 0)
    adjacent_equal = np.all(rgb[:, 1:] == rgb[:, :-1], axis=(2, 3, 4))
    wh = (boxes[..., 2:]-boxes[..., :2])/3
    return dict(past_pixel_change=total, box_pixel_change=inner, outside_box_pixel_change=outer,
        common_pixel_pairs=count, box_pixel_pairs=inner_count, outside_box_pixel_pairs=outer_count,
        identical_adjacent_pairs=adjacent_equal.sum(1), all_eight_images_identical=adjacent_equal.all(1),
        valid_frames=length, mean_coverage=coverage.mean((1, 2, 3))/9,
        box_width_output_pixels=wh[..., 0].mean(1), box_height_output_pixels=wh[..., 1].mean(1),
        box_mask_fraction=mask.mean((1, 2, 3)), temporal_embedding_energy_fraction=fraction,
        identical_embedding_sequence=np.all(embedding[:, 1:] == embedding[:, :-1], axis=(1, 2)))


def describe(values):
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    return dict(rows=len(values), missing=int(len(values)-len(good)),
                mean=float(good.mean()) if len(good) else None,
                quantiles=dict(zip(('min','p10','median','p90','max'),
                    np.quantile(good, [0,.1,.5,.9,1]).tolist())) if len(good) else None)
