"""Label-support partitions for a frozen fitting-only cost diagnosis."""
import numpy as np

from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks, MOMENTS


def support_partitions(prediction, target, selected, valid):
    """Partition evaluation contributions; never return an inference payload."""
    use, mask = np.asarray(selected), np.asarray(valid)
    if (use.dtype != bool or mask.dtype != bool or mask.ndim != 2
            or mask.shape != (len(use), 12) or use.shape != (len(mask),)):
        raise ValueError('Explicit aligned Boolean selection and twelve-step label mask required')
    known = mask.any(1)
    groups = dict(complete=mask.all(1), partial=known & ~mask.all(1), unknown=~known)
    whole = cost_masses(prediction, target, use)
    parts = {name:dict(rows=int((use & take).sum()),
                       masses=cost_masses(prediction, target, use & take))
             for name, take in groups.items()}
    if sum(v['rows'] for v in parts.values()) != int(use.sum()):
        raise ArithmeticError('Label support must exhaust the fixed selected set')
    for name in MOMENTS:
        for field in ('predicted_mass', 'target_mass'):
            total = sum(v['masses'][name][field] for v in parts.values())
            np.testing.assert_allclose(total, whole[name][field], rtol=1e-12, atol=1e-12)
    observed = cost_masses(prediction, target, use & known)
    easy_target = whole['EH']['target_mass']
    unknown_easy = parts['unknown']['masses']['EH']['target_mass']
    return dict(rows=int(use.sum()), parts=parts, whole=whole,
                whole_risks=risks(whole), observed_only_risks=risks(observed),
                unknown_share_of_target_EH=unknown_easy/easy_target if easy_target > 0 else None)
