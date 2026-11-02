"""Post-readout causal-action diagnosis; no labels, new fit or threshold search."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/publication_readiness_2026_09"
HOME = BASE/"european_selection_controller_v1"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(ref):
    path = ROOT/ref["path"]
    if sha(path) != ref["sha256"]: raise ValueError("Frozen causal array or receipt changed")
    return path


def audit():
    chain = json.loads((BASE/"european_selection_forecast_freeze_v1/registration_v2.json").read_text())
    lineage = {r["trial"]:r for r in chain["lineage"] if r["representation"]=="history"}
    bank = json.loads((BASE/"european_selection_forecast_freeze_v1/prediction_freeze.json").read_text())
    physical = np.load(checked(bank["physical_baselines"]),mmap_mode="r",allow_pickle=False)
    forecasts = {}
    for ref in bank["models"]:
        row = json.loads(checked(ref).read_text())
        if row["representation"]=="history": forecasts[row["trial"]]=row["prediction"]
    frozen = json.loads((HOME/"policy_freeze.json").read_text()); rows = []
    for ref in frozen["policies"]:
        cell = json.loads(checked(ref).read_text()); trial=cell["trial"]
        neural=np.load(checked(forecasts[trial]),mmap_mode="r",allow_pickle=False)
        floor=physical[:,lineage[trial]["baseline_index"]]
        legal=np.linalg.norm(neural-floor,axis=-1).max(1)>0
        cost=np.load(checked(cell["prediction"]),mmap_mode="r",allow_pickle=False)[:,6]
        gain=cost[:,0]-cost[:,1]
        all_ok=(cost[:,2]>0)&(cost[:,1]<=.02*cost[:,2])
        easy_ok=(cost[:,3]>0)&(cost[:,4]<=.02*cost[:,3])
        admitted=legal&(gain>0)&all_ok&easy_ok
        choices=np.load(checked(cell["choices"]),allow_pickle=False)
        counts={name:{"selected":int((choice>=0).sum()),"neural":int((choice==6).sum()),
                       "physical":int(((choice>=0)&(choice<6)).sum())}
                for name,choice in zip(cell["policy_names"],choices)}
        rows.append(dict(trial=trial,arm=cell["arm"],rows=len(legal),legal_neural_rows=int(legal.sum()),
            predicted_neural_positive_gain_rows=int((legal&(gain>0)).sum()),
            predicted_neural_all_guard_rows=int((legal&all_ok).sum()),
            predicted_neural_easy_guard_rows=int((legal&easy_ok).sum()),
            predicted_neural_jointly_admitted_rows=int(admitted.sum()),
            predicted_neural_gain_positive_but_guard_rejected_rows=int((legal&(gain>0)&~(all_ok&easy_ok)).sum()),
            policy_choices=counts))
    aggregate={}
    for arm in ("row","context"):
        cells=[r for r in rows if r["arm"]==arm]
        totals={k:sum(r[k] for r in cells) for k in ("rows","legal_neural_rows","predicted_neural_positive_gain_rows",
            "predicted_neural_all_guard_rows","predicted_neural_easy_guard_rows","predicted_neural_jointly_admitted_rows",
            "predicted_neural_gain_positive_but_guard_rejected_rows")}
        totals["policy_choices"]={name:{k:sum(r["policy_choices"][name][k] for r in cells) for k in ("selected","neural","physical")}
                                  for name in cells[0]["policy_choices"]}
        aggregate[arm]=totals
    result=dict(result_source="fresh_post_readout_summary_of_hash_verified_frozen_causal_actions",
        policy_freeze_sha256=sha(HOME/"policy_freeze.json"),cells=rows,aggregate=aggregate,
        repeated_row_views_not_independent_targets=True,selection_label_arrays_read=False,
        future_targets_read=False,new_training=False,threshold_search=False,policy_changed=False,
        scope="causal_admission_volume_not_actual_harm_calibration_or_irreducible_uncertainty")
    out=HOME/"causal_action_diagnosis.json"
    if out.exists() and json.loads(out.read_text())!=result:raise ValueError("Causal diagnosis immutable")
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(aggregate,indent=2,sort_keys=True))


if __name__=="__main__":audit()
