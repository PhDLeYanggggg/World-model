# Shared Reference: Coherent Costs, No Useful Neural Increment

## Completed Experiment

Fresh native arm64 Torch training:54heads,108,000primary optimizer updates,
six frozen contexts,three controller seeds,three candidate banks. The actual
trajectory forecaster stays frozen at seed43. Training PID79115exit0 took
366.151872s,peakRSS10,447,847,424bytes. All25,157-parameter heads match the
parent's initialization,physical-fitting preprocessing and sampled row streams.
Recovery and full2,000-update disabled-sharing controls replay exactly.

All54endpoints were committed at `f1a271e7` before first readout. Readout
PID80142 completes144group views in193.309257s,peakRSS6,562,299,904bytes.
Separate-process replay PID81173 completes exit0 in195.533942s and reproduces
every action hash,metric,summary and the original comparator exactly.
The aggregate-only metric bundle is byte-verified; it contains no source rows
or weights. The combined focused suite passes19tests in2.54s; the full
historical test suite was not rerun.

## Primary Results

All gains below are equal-locality known-label ADE improvements over the
unchanged causal floor. They are not t+50,FDE,metric or independent-test scores.

| Policy | Known-label gain | Conditional95%locality CI |
|---|---:|---:|
| Shared-reference physical-only | 0.553001% | [0.128801,1.286377]% |
| Shared-reference physical+neural | 0.542017% | [0.141372,1.246147]% |
| Shared-reference neural-only | 0.276137% | [0.131230,0.487250]% |

Mixed minus physical-only is **-0.010984pp**,CI[-0.054111,+0.028694].
Its completion-lower paired difference is -0.809107pp,
CI[-2.386719,+0.001198]. The matched-count known difference is -0.009666pp,
CI[-0.051309,+0.027231]. Matched controls measure utility,not safety.

Mixed minus its exact unshared parent is **-0.011770pp**,
CI[-0.066288,+0.034331]. The corresponding completion-lower change is
-0.375313pp,CI[-1.112328,+0.011953]. Neither structural sharing nor a
favorable completion-upper bound demonstrates improved prediction.

The intervals use3,000resamples of12SOURCE localities after context/head-seed
averaging. Contexts,seeds and overlapping rows are not independent recordings.
All SOURCE inner-held outcomes were already exposed; this is conditional
exploratory evidence,not whole-pipeline OOF or independent confirmation.

## Safety Still Fails

The primary mixed policy supports only **21/186** required inner-held
recording/head views and4/72source/head views. It selects23,978row views,
including308unknown outcomes. There are48known all-risk violations,
83known easy-risk violations and20undefined easy-risk views.
The fitting population also fails:84/444recording/head views are supported.

The fixed4xharm sensitivity supports106/186views,with31undefined easy views;
it is not the registered primary policy and is not promoted. Empty selected
risk remains `not_estimable`. No uncertainty interval or aggregate gain
overrides the unchanged2%requirement on every required recording.

## What This Rules Out

Candidate-invariant R/ER targets previously received inconsistent predictions.
The repaired model now enforces exact invariance at every inference chunk,
using the same network and a causal floor token. That implementation invariant
holds,but does not materially improve utility or recording-risk support.
Reference incoherence is therefore not sufficient to explain the failure.

The preceding36-state exact-gradient diagnosis likewise does not support a
generic global-clipping or excessive feasible-support-dual explanation.
Do not infer that a new multiplier schedule or longer threshold search is
the next justified repair. A static scene-context audit is registered instead:
test whether relative motion features discard useful image-region position,
then require a separately controlled training comparison if mapping is usable.

## Boundaries and Reproduction

This is EuropeanSquares oblique webcam detector-track raw image-coordinate
development,obs8/pred12/rawstride12,not SDD annotation ground truth. The
publisher's georeferencing resources are not verified calibration for this
adapter. No seconds,metric,true3D,foundation or deployment success is claimed.
Independent selection,calibration,confirmation and DroneCrowd stay unopened.
Stage5C andSMC remain off.

From the repository root, replay existing frozen endpoints with:

```bash
.venv-pytorch/bin/python scripts/run_m3w_shared_reference_cost.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_shared_reference_cost_v1 --verify
```

Fresh training phases are `register`, `pilot`, `neutral`, `train`, `evaluate`;
completed receipts are immutable and must not be overwritten. Interrupted
phases resume the same roster with `--resume`. Private checkpoints/source
assets are required; a clean public clone alone cannot reconstruct the data.
See [protocol](protocol.md),[data card](data_card.md),[training freeze](training_freeze.json),
[summary](summary.json) and[verification](verification.json).
