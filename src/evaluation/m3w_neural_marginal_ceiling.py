"""Future-informed marginal candidate opportunity; never an inference policy."""
import numpy as np


def candidate_ceiling(ade, floor, allowed, known, hard, scales):
    ade,floor,allowed,known,hard,scales=map(np.asarray,(ade,floor,allowed,known,hard,scales))
    n=len(floor)
    if (ade.shape!=(n,7) or allowed.shape!=ade.shape or allowed.dtype!=bool
            or known.shape!=(n,) or known.dtype!=bool or hard.shape!=(n,) or hard.dtype!=bool
            or scales.shape!=(n,) or (scales<=0).any() or not np.isfinite(scales).all()
            or not np.isfinite(ade[known]).all() or not np.isfinite(floor[known]).all()
            or (ade[known]<0).any() or (floor[known]<0).any()):
        raise ValueError('Seven aligned known-evaluation costs and causal eligibility required')
    physical=np.minimum(floor,np.min(np.where(allowed[:,:6],ade[:,:6],np.inf),axis=1))
    mixed=np.minimum(physical,np.where(allowed[:,6],ade[:,6],np.inf))
    benefit=physical-mixed
    def subset(mask):
        take=known & mask;reference=float(floor[take].sum())
        ratio=lambda x:float(100*x/reference) if reference>0 else None
        return dict(known_rows=int(take.sum()),floor_mass=reference,
            physical_oracle_gain_floor_percent=ratio(float((floor[take]-physical[take]).sum())),
            mixed_oracle_gain_floor_percent=ratio(float((floor[take]-mixed[take]).sum())),
            neural_marginal_gain_floor_pp=ratio(float(benefit[take].sum())),
            neural_strictly_better_rows=int((benefit[take]>0).sum()),
            available_neural_rows=int((allowed[:,6]&take).sum()),
            normalized_neural_gain_quantiles=np.quantile(benefit[take]/scales[take],[0,.5,.9,.99,1]).tolist() if take.any() else None)
    return dict(all=subset(np.ones(n,bool)),hard=subset(hard),unknown_rows=int((~known).sum()),
        oracle_uses_future_labels=True,oracle_is_deployable=False,
        labels_exported_as_inference_inputs=False,scope='known_ADE_not_unobserved_full_path_or_joint_safety')
