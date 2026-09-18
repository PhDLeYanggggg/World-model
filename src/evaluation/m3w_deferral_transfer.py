"""Fixed source-site cost-deferral readout; no policy fitting or calibration."""
import numpy as np

from src.evaluation.m3w_recording_diagnostic import error_summary, paired_gain_interval


def outputs(proposal, score=None):
    proposal = np.asarray(proposal, float)
    if proposal.ndim != 3 or proposal.shape[1:] != (12, 2) or not np.isfinite(proposal).all():
        raise ValueError('Finite twelve-step candidate required')
    result = {'proposal': proposal}
    if score is not None:
        score = np.asarray(score)
        if score.shape != (len(proposal),) or not np.isfinite(score).all():
            raise ValueError('One finite observed-input score per candidate required')
        result['hard_action'] = np.where((score > 0)[:, None, None], proposal, 0.)
    return result


def errors(prediction, target, requested):
    if prediction.shape != target.shape or requested.shape != (len(target),):
        raise ValueError('Aligned predictions, labels and request masks required')
    return dict(ade=np.linalg.norm(prediction-target, axis=-1).mean(1),
                fde=np.linalg.norm(prediction[:, -1]-target[:, -1], axis=-1),
                changed=np.any(prediction != 0, axis=(1, 2)).astype(float),
                requested=requested.astype(float))


def seed_mean(cells):
    if len(cells) != 3 or any(set(c) != set(cells[0]) for c in cells):
        raise ValueError('Exactly three matched seed-error records required')
    for field in cells[0]:
        if any(np.asarray(c[field]).shape != np.asarray(cells[0][field]).shape for c in cells):
            raise ValueError('Seed rows differ')
    return {field:np.mean([c[field] for c in cells], axis=0) for field in cells[0]}


def grouped_summary(values, baseline, native, hard, groups, inverse, counts):
    result = error_summary(values['ade'], values['fde'], baseline, native, values['changed'], hard)
    result['requested_rate'] = float(values['requested'].mean())
    result['gain_interval'] = paired_gain_interval(baseline, values['ade'], baseline, inverse, counts)
    result['hard_gain_interval'] = paired_gain_interval(baseline, values['ade'], baseline, inverse, counts, hard)
    records = []
    for kind, labels in groups.items():
        ade, cv, gains = [], [], []
        for label in sorted(set(labels)):
            mask = labels == label
            row = error_summary(values['ade'][mask], values['fde'][mask], baseline[mask],
                                native[mask], values['changed'][mask], hard[mask])
            ade.append(row['ade']); cv.append(row['cv_ade'])
            if kind == 'recording':
                records.append(dict(recording=str(label), **row))
                if row['gain_percent'] is not None:
                    gains.append(row['gain_percent'])
        result[f'equal_{kind}_gain_percent'] = float(100*(1-np.mean(ade)/np.mean(cv))) if np.mean(cv) > 0 else None
        if kind == 'recording':
            result['worst_recording_gain_percent'] = min(gains) if gains else None
    return result, records
