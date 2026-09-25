"""Source-only producer and calibration roles, never confirmation roles."""
import hashlib

def inner_halves(folds, counts, salt):
    if set(folds) != set(counts) or set(folds.values()) != {0, 1, 2}:
        raise ValueError('Complete frozen three-fold source roster required')
    result = {}
    for fold in range(3):
        sites = [s for s, f in folds.items() if f == fold]
        if len(sites) != 4 or any(counts[s] <= 0 for s in sites):
            raise ValueError('Four supported source sites per fold required')
        ordered = sorted(sites, key=lambda s: (-counts[s], s))
        parts = [[], []]
        for offset in (0, 2):
            pair = sorted(ordered[offset:offset+2], key=lambda s:
                hashlib.sha256((salt+'|'+s).encode()).hexdigest())
            for side, site in enumerate(pair):
                parts[side].append(site)
        result[str(fold)] = [sorted(p) for p in parts]
    return result


def role_sets(folds, fit_fold, calibration_fold):
    if fit_fold not in range(3) or calibration_fold not in range(3) or fit_fold == calibration_fold:
        raise ValueError('Distinct declared fitting and calibration folds required')
    outer = ({0, 1, 2}-{fit_fold, calibration_fold}).pop()
    return dict(fitting=sorted(s for s, f in folds.items() if f == fit_fold),
        calibration=sorted(s for s, f in folds.items() if f == calibration_fold),
        readout=sorted(s for s, f in folds.items() if f == outer))


def check_producer_roles(trained, predicted, calibration, readout):
    sets = [set(x) for x in (trained, predicted, calibration, readout)]
    if any(not x for x in sets) or any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        raise ValueError('Every nested producer role must be disjoint')
    return True

