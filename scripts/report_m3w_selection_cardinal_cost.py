"""Public-scalar reconstruction; optional private checkpoint trace export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_selection_cardinal_cost import summarize

HOME=ROOT/"outputs/publication_readiness_2026_09/european_selection_cardinal_cost_v1"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def public(ref,subdirectory):
    path=ROOT/ref["path"]
    if path.parent!=HOME/subdirectory or sha(path)!=ref["sha256"]:raise ValueError("Public scalar identity changed")
    return json.loads(path.read_text())


def verified():
    cfg=json.loads((ROOT/"configs/m3w_selection_cardinal_cost_v1.json").read_text())
    complete=json.loads((HOME/"complete.json").read_text());v=json.loads((HOME/"verification.json").read_text())
    s=json.loads((HOME/"summary.json").read_text());training=json.loads((HOME/"training_freeze.json").read_text())
    if (v["summary_sha256"]!=sha(HOME/"summary.json") or v["complete_sha256"]!=sha(HOME/"complete.json")
            or not v["all27_new_inferences_exact"] or not v["all54_metric_groups_exact"] or not v["cached_control_exact"]):
        raise ValueError("Separate inference and metric replay required")
    rows=[public(ref,"groups") for ref in complete["groups"]];fits=[public(ref,"fits") for ref in training["fits"]]
    if len(rows)!=54 or len(fits)!=27 or summarize(rows,cfg)!=s:raise ValueError("Complete scalar reconstruction differs")
    return s,rows,fits,training,complete,v


def export_traces():
    from src.world_model.m3w_causal_descriptor_head import read_checkpoint
    training=json.loads((HOME/"training_freeze.json").read_text());values=[]
    for ref in training["fits"]:
        fit=public(ref,"fits");path=ROOT/fit["checkpoint"]["path"]
        if sha(path)!=fit["checkpoint"]["sha256"]:raise ValueError("Frozen private weight identity changed")
        cp=read_checkpoint(path)
        if cp["step"]!=cp["budget"] or cp["unknown_supervised_rows"]:raise ValueError("Complete SOURCE fit required")
        values.append(dict(trial=fit["trial"],arm=fit["arm"],seed=fit["seed"],parameters=fit["parameters"],
            checkpoint_sha256=fit["checkpoint"]["sha256"],trace=cp["trace"],source_only=True))
    if len(values)!=27:raise ValueError("All27fits required")
    value=dict(result_source="cached_verified_scalar_export_of_fresh_SOURCE_training",models=values,
        training_freeze_sha256=sha(HOME/"training_freeze.json"),labels_or_weights_exported=False,new_training=False)
    path=HOME/"training_traces.json"
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError("Trace export immutable")
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    return dict(models=27,scalar_trace_points=sum(len(v["trace"]) for v in values))


def render():
    s,rows,fits,t,c,v=verified();ci=lambda x:f"{x['point']:+.4f} [{x['CI95'][0]:+.4f}, {x['CI95'][1]:+.4f}]"
    table=[]
    for arm in s["accuracy"]:
        risk=s["safety"][arm];rate=100*sum(r["policies"][arm]["switch_rate"] for r in rows)/len(rows)
        table.append(f"|{arm}|{ci(s['accuracy'][arm])}|{rate:.3f}%|{risk['worst_known_easy_net_degradation_percent']:.3f}%|"
            f"{risk['supported_record_views']}/{risk['required_record_views']}|{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']}|{risk['undefined_easy_views']}|")
    contrast=[f"|{a}|{ci(x['known_pp'])}|{ci(x['completion_lower_pp'])}|" for a,x in s["effects"].items()]
    bysite=[]
    for site in sorted(s["effects"]["mean_absolute"]["known_pp"]["by_locality"]):
        x=s["effects"]["mean_absolute"]
        bysite.append(f"|{site}|{x['known_pp']['by_locality'][site]:+.4f}|{x['completion_lower_pp']['by_locality'][site]:+.4f}|")
    losses=[]
    for arm in ("mean_centered","weighted_absolute","mean_absolute"):
        cells=[f for f in fits if f["arm"]==arm]
        losses.append(f"|{arm}|{len(cells)}|{sum(f['step'] for f in cells)}|"
            f"{sum(f['first_trace']['total'] for f in cells)/len(cells):.5f}|{sum(f['final_trace']['total'] for f in cells)/len(cells):.5f}|")
    text=f"""# Cardinal Cost-Loss Repair: Complete Readout

## Scope And Result Sources

Fresh_run:27complete Torch SOURCE cost fits,three separate complete first-producer
refits,new causal cost inference,27pointwise policy freezes,and54selection metric
groups. Cached_verified:the nine original context critics,the original fixed
forecast bank, SOURCE rows and exposed selection labels. No new forecasting
model,data split,normalization,feature,architecture or risk threshold is introduced.
The six selection localities are already exposed development data,not final test.
Calibration/confirmation/DroneCrowd remain closed. SOURCE resubstitution is not validation.

All arms share29,253parameters,initial weights,complete query draws and original
producer-excluded eight SOURCE sites. The forecaster fits the other four sites.
Each source query has at least four visits. Unknown outcomes are not supervised
zeros. Inference accepts only the original444causal fields and causal rollouts.

## Controlled Factors

Original weighted_centered has H/EH underprediction multiplier4 and within-row
centered net-gain-error loss. mean_centered changes only multiplier4to1;
weighted_absolute changes only centered to uncentered net-gain error;
mean_absolute combines both. The latter is the registered primary,not selected
after looking at outcomes. Same decoder shares R/ER and keeps B<=R,ER<=R,EH<=H
and B+H<=causal disagreement. Removing asymmetry does not guarantee correct
conditional means in this clipped,finite-capacity,multi-objective model.
Absolute gain anchoring is not a new calibration guarantee or a new dynamics model.
The historical144-fit easy-deviance failure remains a distinct negative result.

## Registered Primary And Factor Controls

All contrasts versus the frozen weighted_centered context independent policy;
units are percentage points relative to each producer's SOURCE-selected floor.

|New loss arm minus original|Known-label secondary [95% CI]|Paired completion-lower [95% CI]|
|---|---:|---:|
{chr(10).join(contrast)}

Primary pass:**{s['primary_pass']}**. Full-policy original record-risk pass:
**{s['original_record_risk_pass']}**. Candidate:**{s['policy_candidate_pass']}**.
Deployment changed:**false**. Positive secondary cannot replace the primary or
the unchanged all/easy positive-harm requirements. No joint solver is retrained
or reoptimized; the earlier failed matched-joint scientific result remains.

## Accuracy, Intervention And Safety

|Policy|Known ADE gain vs floor,% [CI]|Mean switch rate|Worst known easy net degradation|Supported repeated record views|Known all/easy risk violations|Undefined easy-risk views|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All38,102targets,28recordings,six localities remain,including futureless and
partial outcomes. There are252repeated recording-risk views,not252independent
recordings. The2%budget applies to every required record's positive all/easy
harm and easy-net degradation. Benefits cannot cancel positive-harm numerators.
An empty/zero-reference easy-risk view is not estimable,not a safety pass.
No favorable subset or seed is removed. These are available-label ADE scores;
full12step FDE is not available on the entire roster. Endpoint scores are diagnostic.

## Primary By Locality

|Selection locality|Known component,pp|Paired completion lower,pp|
|---|---:|---:|
{chr(10).join(bysite)}

Intervals:3000locality bootstrap draws,after averaging three SOURCE folds and
actual seeds17/29/43 inside each locality. Overlapping windows and repeated
producers are not independent samples. This is conditional development evidence,
not calibrated population safety or untouched final confirmation.

## Actual Training Loss And Runtime

|Loss arm|Complete new fits|Primary updates|Mean first minibatch total|Mean final minibatch total|
|---|---:|---:|---:|---:|
{chr(10).join(losses)}

Loss definitions differ by design; cross-arm raw training totals are not a
common test metric. First/final minibatches differ and are not validation loss.
Scalar per-update traces are exported separately from hash-verified checkpoints.
[All real fitting traces](figures/training_traces.svg) and
[paired utility and easy-net risk](figures/utility_and_easy_risk.svg) are derived
only from these public scalar artifacts,not new experiments.
No weights or labels are published. The original control100-step weights,
optimizer,sampler,trace and RNG match the previous real pilot exactly. Every
new arm passes50->100resume vs100direct; three complete first-producer refits
also match exactly. These are engineering reproducibility checks,not independent
scientific confirmation.

Formal train PID{t['pid']}:{t['seconds']:.6f}s,peak{t['peak_RSS_bytes']}B;
primary updates{t['primary_optimizer_updates']},complete-refit updates{t['exact_full_refit_updates']}.
Evaluation PID{c['pid']}:{c['seconds']:.6f}s. Separate replay PID{v['pid']}:
{v['seconds']:.6f}s. All27new cost inferences,choices and54scalar metric groups
reproduce exactly; cached original control identity remains exact. Local arm64
CPU4/inter-op1/workers0. No CREATE job or remote scheduler claim.

## Interpretation Boundary

The result tests a specific SOURCE objective repair,not new JEPA/Transformer
representation or trajectory learning. Utility,positive harm,unknown outcomes
and support must be read together. Risk calibration and a useful safe joint
policy remain separate requirements before final confirmation. Do not deploy a
better mean with unsafe recordings or call a zero-switch fallback learned success.

obs8/pred12/rawstride12,local-image pixels,detector-silver,2.5D/pseudo-3D agent
trajectory state modeling. Not metric,seconds,human gold,true3D or foundation
evidence. No Stage5C/SMC. Submission readiness remains false.
"""
    (HOME/"readout.md").write_text(text)
    gates=f"""# Cardinal Cost-Loss Gates

- All27SOURCE-only complete fits and exact coverage:true.
- Original control100step and all three complete first-producer refits exact:true.
- All27new inference/54metric-group replay:true.
- Registered primary CI lower>0:{s['primary_pass']}.
- Original complete required-record risk:{s['original_record_risk_pass']}.
- Policy candidate:{s['policy_candidate_pass']}.
- Independent calibration/final confirmation:not_run,still closed.
- Joint allocation contribution:not established by this test;old failure retained.
- Deployment changed:false. Submission ready:false. Stage5C:false. SMC:false.

Engineering completion is not scientific success. Undefined selected-easy risk
is not a safety pass. Read all arms and negative outcomes in readout.md.
"""
    (HOME/"gates.md").write_text(gates)
    receipt=dict(result_source="cached_verified_public_scalar_reconstruction_of_fresh_training_and_eval",
        summary_sha256=sha(HOME/"summary.json"),readout_sha256=sha(HOME/"readout.md"),
        private_input_reads=0,primary_pass=s["primary_pass"],original_record_risk_pass=s["original_record_risk_pass"],
        deployment_changed=False,submission_ready=False)
    (HOME/"report_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--export-traces",action="store_true");args=parser.parse_args()
    print(json.dumps(export_traces() if args.export_traces else render()))
