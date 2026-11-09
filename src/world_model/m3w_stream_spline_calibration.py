"""Site-separated controller calibration with the original recording scorer."""
import numpy as np


def require_roles(identity, sites, omitted, seed):
    exclude = set(omitted)
    if (not exclude <= set(sites) or tuple(identity['omitted']) != tuple(omitted)
            or set(identity['fit']) != set(sites)-exclude or identity['seed'] != seed):
        raise ValueError('Controller lineage does not match the requested exclusions')


def select_recording_rule(evaluated):
    def rates(cells):
        return np.array([np.mean([(r['dual_positive']/r['views'], r['risk_pass']/r['views'], r['vs_cv'])
                                  for r in cells[s]], axis=0) for s in sorted(cells)])

    ref = rates(evaluated[None])
    table = []
    for rule, cells in evaluated.items():
        if sorted(cells) != sorted(evaluated[None]):
            raise ValueError('Candidate changed the calibration locations')
        score = rates(cells)
        v = score.mean(0)
        ok = (score[:, :2] >= ref[:, :2]-1e-12).all() and v[2] >= ref[:, 2].mean()-1e-12
        table.append(dict(rule=rule, dual=float(v[0]), risk=float(v[1]), utility=float(v[2]), admissible=bool(ok)))
    best = max((r for r in table if r['admissible']), key=lambda r: (r['dual'], r['risk'], r['utility']))
    return best['rule'], table
