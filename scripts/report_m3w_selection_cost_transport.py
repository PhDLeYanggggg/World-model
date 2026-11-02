"""Reconstruct a frozen diagnostic report from public scalar artifacts only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_selection_cost_transport import pooled

HOME=ROOT/"outputs/publication_readiness_2026_09/european_selection_cost_transport_v1"


def checked(ref):
    path=ROOT/ref["path"]
    if not path.is_relative_to(ROOT/"outputs") or hashlib.sha256(path.read_bytes()).hexdigest()!=ref["sha256"]:
        raise ValueError("Public scalar identity required")
    return json.loads(path.read_text())


def verified():
    complete=json.loads((HOME/"complete.json").read_text());verification=json.loads((HOME/"verification.json").read_text())
    rows=[checked(ref) for ref in complete["groups"]];summary=json.loads((HOME/"summary.json").read_text())
    if len(rows)!=252 or pooled(rows)!=summary:raise ValueError("All scalar groups required")
    for file,key in (("summary.json","summary_sha256"),("complete.json","complete_sha256")):
        if hashlib.sha256((HOME/file).read_bytes()).hexdigest()!=verification[key]:raise ValueError("Verification binding differs")
    if not verification["all18_SOURCE_cost_inferences_exact"] or not verification["all252_diagnostic_groups_exact"]:
        raise ValueError("Separate numerical replay required")
    return summary,complete,verification


def render():
    s,c,v=verified();d=s["pooled_descriptive_action_view_diagnostics"]
    lines=["# Frozen Cost Transport Diagnosis","",
        "All144SOURCE fitting-resubstitution and108exposed-selection groups are complete.",
        "This is a read-only outcome diagnosis of18frozen critics,not a new policy or calibration.",
        "All native masses below pool repeated action views; they are descriptive,not independent samples.","",
        "|Domain / arm, selected actions|Predicted all risk|Observed all risk|Predicted easy risk|Observed easy risk|R inflation|EH predicted/actual|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for domain in ("source_fit","selection"):
        for arm in ("row","context"):
            a=d[domain][arm]["subsets"]["selected"]
            lines.append(f"|{domain} / {arm}|{100*a['predicted_all_risk_on_known']:.3f}%|{100*a['observed_all_risk_on_known']:.3f}%|{100*a['predicted_easy_risk_on_known']:.3f}%|{100*a['observed_easy_risk_on_known']:.3f}%|{a['reference_inflation_ratio']:.3f}x|{a['easy_harm_prediction_ratio']:.4f}|")
    lines += ["","## Interpretation","",
        "The selected-action risk failure already exists on fitting SOURCE data: it is not solely a new-scene shift.",
        "For context selection actions,the all-risk budget error has harm-underprediction mass2758.86 and",
        "reference-inflation contribution5702.98; the denominator error is substantial. Easy-risk error is",
        "mostly harm underprediction:1182.87 versus reference contribution-39.37. These signed components",
        "have native pixel-cost units and do not establish an individual causal mechanism.","",
        "All-legal neural actions show aggregate harm overprediction,whereas selected actions show harm",
        "underprediction. Selection changes which errors matter; a single global correction factor is not justified.",
        "The R/ER outputs are already shared across candidate actions,so this is not an absent shared-reference implementation.","",
        "## Useful Neural Opportunity Funnel","",
        "|Context domain|Useful known row views|Gain mass lost at nonpositive predicted gain|Gain mass guard-rejected|Gain mass actually neural-chosen|",
        "|---|---:|---:|---:|---:|"]
    for domain in ("source_fit","selection"):
        a=d[domain]["context"];f=a["neural_funnel"];total=a["useful_known_neural_benefit_mass"]
        frac=lambda k:100*f[k]["known_benefit_mass"]/total
        lines.append(f"|{domain}|{a['useful_known_neural_rows']}|{frac('predicted_nonpositive_gain'):.2f}%|{frac('predicted_positive_guard_rejected'):.2f}%|{frac('neural_chosen'):.2f}%|")
    lines += ["","The true useful designation uses outcomes only for this diagnosis; it is never an inference feature.",
        "Unknown outcomes remain unknown. Known beneficial gain does not certify safety.","",
        "## Next Controlled Test","",
        "Keep forecasts,model,data,normalization and2%rules fixed. Compare the original asymmetric/centered",
        "objective with unweighted squared mean-cost loss,absolute net-gain anchoring,and their combination.",
        "This tests specific optimization hypotheses,not a promised cure. Historical easy-deviance144fits",
        "already failed and must not be repeated or relabeled as a new method. Risk calibration and final",
        "confirmation remain closed; the six selection sites are already exposed development data.","",
        "## Execution And Scope","",
        f"Fresh diagnostic run:PID{c['pid']},{c['seconds']:.6f}s,peak{c['peak_RSS_bytes']}B.",
        f"Separate replay:PID{v['pid']},{v['seconds']:.6f}s,peak{v['peak_RSS_bytes']}B.",
        "All18SOURCE model inferences and252diagnostic groups reproduce exactly. Selection predictions are",
        "cached_verified; no claim of newly generated selection predictions. Frozen actions unchanged; no solver rerun.",
        "No new training,threshold search,normalization,calibration or deployment. Original failed primary/risk unchanged.",
        "obs8/pred12/rawstride12,local-image pixels,detector-silver,available labels. Not metric,seconds,human gold,",
        "true3D or foundation evidence. Stage5C/SMC off.","",
        "## Reproduction","","```sh",
        ".venv-pytorch/bin/python scripts/run_m3w_selection_cost_transport.py register",
        ".venv-pytorch/bin/python scripts/run_m3w_selection_cost_transport.py pilot",
        ".venv-pytorch/bin/python scripts/run_m3w_selection_cost_transport.py run --resume",
        ".venv-pytorch/bin/python scripts/run_m3w_selection_cost_transport.py verify",
        ".venv-pytorch/bin/python scripts/report_m3w_selection_cost_transport.py","```","",
        "Completed phase endpoints are immutable; commands refuse overwrites. Same-contract partial execution",
        "resumes from case checkpoints. Private data/model receipts are needed for inference,not scalar reconstruction.",""]
    (HOME/"readout.md").write_text("\n".join(lines))
    return dict(groups=252,private_input_reads=0,new_training=False,policy_changed=False)


if __name__=="__main__":print(json.dumps(render()))
