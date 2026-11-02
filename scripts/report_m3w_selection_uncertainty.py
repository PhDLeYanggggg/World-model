"""Render the registered seed-spread comparison using public scalar evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_selection_uncertainty import summarize

HOME=ROOT/"outputs/publication_readiness_2026_09/european_selection_uncertainty_v1"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def public(ref,subdir):
    p=ROOT/ref["path"]
    if p.parent!=HOME/subdir or sha(p)!=ref["sha256"]:raise ValueError("Public scalar identity changed")
    return json.loads(p.read_text())


def verified():
    cfg=json.loads((ROOT/"configs/m3w_selection_uncertainty_v1.json").read_text())
    complete=json.loads((HOME/"complete.json").read_text());v=json.loads((HOME/"verification.json").read_text())
    s=json.loads((HOME/"summary.json").read_text());t=json.loads((HOME/"training_freeze.json").read_text())
    if (v["summary_sha256"]!=sha(HOME/"summary.json") or v["complete_sha256"]!=sha(HOME/"complete.json")
            or not v["all27_new_inferences_exact"] or not v["all54_metric_groups_exact"] or not v["original_disabled_control_exact"]):
        raise ValueError("Independent numerical replay required")
    rows=[public(r,"groups") for r in complete["groups"]];fits=[public(r,"fits") for r in t["fits"]]
    if len(rows)!=54 or len(fits)!=27 or summarize(rows,cfg)!=s:raise ValueError("Complete scalar reconstruction differs")
    return s,rows,fits,t,complete,v


def export_traces():
    from src.world_model.m3w_causal_descriptor_head import read_checkpoint
    training=json.loads((HOME/"training_freeze.json").read_text());rows=[]
    for ref in training["fits"]:
        fit=public(ref,"fits");p=ROOT/fit["checkpoint"]["path"]
        if sha(p)!=fit["checkpoint"]["sha256"]:raise ValueError("Frozen SOURCE checkpoint changed")
        cp=read_checkpoint(p)
        if cp["step"]!=cp["budget"] or cp["unknown_supervised_rows"]:raise ValueError("Full known-only SOURCE fit required")
        rows.append(dict(trial=fit["trial"],arm=fit["arm"],seed=fit["seed"],parameters=fit["parameters"],
            checkpoint_sha256=fit["checkpoint"]["sha256"],trace=cp["trace"],source_only=True))
    if len(rows)!=27:raise ValueError("All27SOURCE fits required")
    value=dict(result_source="cached_verified_scalar_export_of_fresh_SOURCE_training",models=rows,
        training_freeze_sha256=sha(HOME/"training_freeze.json"),labels_or_weights_exported=False,new_training=False)
    p=HOME/"training_traces.json"
    if p.exists() and json.loads(p.read_text())!=value:raise ValueError("Scalar traces immutable")
    p.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    return dict(models=27,scalar_trace_points=sum(len(r["trace"]) for r in rows))


def render():
    s,rows,fits,t,c,v=verified();ci=lambda x:f"{x['point']:+.4f} [{x['CI95'][0]:+.4f}, {x['CI95'][1]:+.4f}]"
    table=[]
    for arm in ("disabled","real","noise"):
        risk=s["safety"][arm];switch=100*sum(r["policies"][arm]["switch_rate"] for r in rows)/len(rows)
        table.append(f"|{arm}|{ci(s['accuracy'][arm])}|{switch:.3f}%|{risk['worst_known_easy_net_degradation_percent']:.3f}%|"
            f"{risk['supported_record_views']}/{risk['required_record_views']}|{risk['known_all_risk_violations']}/{risk['known_easy_risk_violations']}|{risk['undefined_easy_views']}|")
    effects=[f"|{name}|{ci(a['known_pp'])}|{ci(a['completion_lower_pp'])}|{ci(a['completion_upper_pp'])}|" for name,a in s["effects"].items()]
    seeds=[]
    for seed in (17,29,43):
        cells=[r for r in rows if r["seed"]==seed]
        seeds.append(f"|{seed}|{sum(r['effects']['real_minus_disabled']['known_pp'] for r in cells)/len(cells):+.4f}|"
            f"{sum(r['effects']['real_minus_disabled']['completion_lower_pp'] for r in cells)/len(cells):+.4f}|")
    sites=[]
    known=s["effects"]["real_minus_disabled"]["known_pp"]["by_locality"]
    lower=s["effects"]["real_minus_disabled"]["completion_lower_pp"]["by_locality"]
    for site in sorted(known):sites.append(f"|{site}|{known[site]:+.4f}|{lower[site]:+.4f}|")
    losses=[]
    for arm in ("disabled","real","noise"):
        cells=[r for r in fits if r["arm"]==arm]
        losses.append(f"|{arm}|{len(cells)}|{sum(r['step'] for r in cells)}|"
            f"{sum(r['first_trace']['total'] for r in cells)/len(cells):.5f}|{sum(r['final_trace']['total'] for r in cells)/len(cells):.5f}|")
    content=f"""# Seed Disagreement For Safe Intervention: Complete Readout

## Material Passport

Mode:run and numerical reproducibility validation. Scope:SOURCE-trained critic
input comparison on six exposed model-selection localities. Not independent
confirmation,new forecasting/dynamics,calibration guarantee or deployment.

## Result Sources And Method

Fresh_run:six derived spread feature arrays and complete reconstruction,27real
Torch cost fits,three complete first-producer refits,27causal pointwise action
freezes,54selection metric groups and a separate inference/metric replay.
Cached_verified:nine actual same-fold forecasters,original fixed seven candidate
paths,149,510row SOURCE whitelist,selection features and exposed evaluation labels.
No new trajectory predictor,ensemble-mean forecast action or target filtering.

Three actual seeds17/29/43 train on the same four SOURCE producer sites. All
three exclude each critic's eight fitting sites;SOURCE cache IDs and order match.
On selection,all are frozen SOURCE-trained models. Derive16fields from predicted
relative paths only:12radial spreads,max pair path difference,direction dispersion,
step-speed spread and path-length spread. Observed future labels/endpoints,
future availability,central velocity and test goal statistics never enter inputs.

The old444field query-context critic has a zero-initialized16->64biasless
projection. All three arms share30,277parameters,original initial weights,RNG,
normalization,complete query draws and fixed mean_absolute loss. disabled has
no added input;real uses seed spread;noise draws deterministic row-ID-hashed
covariates from the fitting-SOURCE-only balanced8x512catalogue. This noise is
a fitted random prior,not an observed current-query modality or selection statistic.
Real and noise use the same SOURCE mean/std and clip8. No outcome fits that prior.
Every full disabled model,base optimizer state,trace and prediction matches the
previous mean_absolute control exactly. Extra parameters alone do not explain
a real-versus-noise contrast,but this is still a limited input experiment.

The original expected gain>0,all/easy positive-harm2%guards and shared R/ER
constraints remain. All27policies freeze before this comparison scores labels.
No threshold search,joint solver or test-driven normalization. Ensemble spread
is established methodology,not new M3W theory or a calibrated confidence interval.
See the primary-source references and hypothesis limitations in protocol.md.

## Registered Primary And Negative Control

Paired effects in percentage points relative to each producer's fixed SOURCE
floor. Primary:real-minus-disabled paired completion lower. Negative control:
real-minus-noise. Known effects are secondary,not replacements for the primary.

|Contrast|Known secondary [95% CI]|Paired completion lower [95% CI]|Completion upper diagnostic [95% CI]|
|---|---:|---:|---:|
{chr(10).join(effects)}

Primary pass:**{s['primary_pass']}**. Negative-control pass:**{s['negative_control_pass']}**.
Original full-record risk pass:**{s['original_record_risk_pass']}**.
Policy candidate:**{s['policy_candidate_pass']}**. Deployment changed:**false**.
The earlier failed matched-joint result is not replaced by this independent-arm test.

## Utility And Required Recording Risk

|Arm|Known ADE gain vs floor,% [95% CI]|Mean intervention|Worst known easy net degradation|Supported repeated record views|Known all/easy risk violations|Undefined easy-risk views|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All38,102targets,28recordings and six localities remain. There are37,254known
outcomes,including15,820partial futures and21,434complete12step targets;
848wholly unknown outcomes remain. Endpoint labels exist on27,694rows only.
ADE uses available steps,not a complete12step-only subset. Endpoint FDE is
diagnostic;full-roster endpoint error is not identified.

252recording-risk views repeat the28recordings over nine producer models;they
are not252independent recordings. Positive harm cannot be canceled by benefit.
Every required record retains original2%all/easy positive-harm and easy-net
requirements. Empty selected-easy reference is not estimable,not safe zero.
No favorable locality,seed,partial target or undefined risk view is hidden.

## Locality And Seed Effects

|Locality|Real-minus-disabled known,pp|Paired completion lower,pp|
|---|---:|---:|
{chr(10).join(sites)}

|Actual model seed|Real-minus-disabled known,pp|Paired completion lower,pp|
|---|---:|---:|
{chr(10).join(seeds)}

3000paired locality bootstrap draws average SOURCE folds/seeds inside each of
six localities first. Overlapping windows are not independent observations.
This is conditional model-selection development evidence,not a population
safety certificate or an untouched final confirmation interval. The six sites
were already exposed before this experiment. Calibration/confirmation and
DroneCrowd remain closed. SOURCE fitting error is resubstitution,not validation.

## Actual Training And Runtime

|Arm|Complete fits|Primary updates|Mean first minibatch loss|Mean final minibatch loss|
|---|---:|---:|---:|---:|
{chr(10).join(losses)}

All arms share the same loss,but first/final minibatches differ and are not
validation errors. [All27real traces](figures/training_traces.svg) and
[paired utility and easy-risk summary](figures/utility_and_risk.svg) are derived
from public scalar traces,not new experiments. Unknown futures have no supervised
zero label. Each original SOURCE query has at least four visits. All three
50->100recovery pilots and all three complete first-producer refits are exact.

Formal training PID{t['pid']}:{t['seconds']:.6f}s,peak{t['peak_RSS_bytes']}B;
primary updates{t['primary_optimizer_updates']},complete-refit updates{t['exact_full_refit_updates']}.
Evaluation PID{c['pid']}:{c['seconds']:.6f}s. Separate replay PID{v['pid']}:
{v['seconds']:.6f}s. All27new inferences/choices and54metric groups reproduce
exactly. Disabled controls remain identical to the preceding experiment.
Native arm64CPU4/inter-op1/workers0,no NumPy training fallback. No CREATE job
was submitted for this comparison;remote scheduler state is not inferred.

## Limits And Remaining Paper Requirements

Three same-data initializations can share blind spots and fail under scene
shift. Point-forecast spread omits predictive-likelihood and irreducible outcome
uncertainty. It is a possible risk feature,not a coverage guarantee. Generic
AUROC,cost MSE,lower loss or a positive secondary cannot establish safe switches.
No pairwise collision/composition optimization or new dynamics was tested.

A useful safe full policy,matched joint controls,appropriate public forecasting
baselines,independent risk calibration/final confirmation and anonymous end-to-
end reproduction remain necessary. Old main PDF is not regenerated for this
readout. No deployment or submission-readiness promotion follows from completion.

obs8/pred12/rawstride12,local-image pixel,detector-silver,2.5D/pseudo-3D trajectory
world state. Not old rawt50,metric,seconds,human gold,true3D or foundation.
Stage5C not executed;SMC not enabled.
"""
    (HOME/"readout.md").write_text(content)
    (HOME/"gates.md").write_text(f"""# Seed-Spread Input Gates

- Complete six causal feature arrays and independent replay:true.
- All27full SOURCE fits,three exact full refits and9exact old controls:true.
- All27new inference/54metric-group replay:true.
- Registered real-minus-disabled paired-lower CI lower>0:{s['primary_pass']}.
- Real-minus-noise paired-lower CI lower>0:{s['negative_control_pass']}.
- Original complete required-record risk:{s['original_record_risk_pass']}.
- Policy candidate:{s['policy_candidate_pass']}.
- Independent calibration/final confirmation:not_run,still closed.
- Joint contribution:not established by this experiment.
- Deployment changed:false. Submission ready:false. Stage5C:false. SMC:false.

Undefined selected-easy risk is not safe zero. Engineering completion does not
establish the scientific hypothesis;read all arms,negative outcomes and limits.
""")
    receipt=dict(result_source="cached_verified_public_scalar_reconstruction_of_fresh_training_and_eval",
        summary_sha256=sha(HOME/"summary.json"),readout_sha256=sha(HOME/"readout.md"),private_input_reads=0,
        primary_pass=s["primary_pass"],negative_control_pass=s["negative_control_pass"],original_record_risk_pass=s["original_record_risk_pass"],
        deployment_changed=False,submission_ready=False)
    (HOME/"report_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--export-traces",action="store_true");args=parser.parse_args()
    print(json.dumps(export_traces() if args.export_traces else render()))
