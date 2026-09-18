# Frozen Deferral Readout: Training Benefit Does Not Transfer

## Result

The six fixed neural endpoints were evaluated without new training, checkpoint
selection or threshold search. Direct cost supervision reduces damage relative
to the losing dense neural control, but still loses to stationary constant
velocity (CV). The expected-cost gate rejects every query. Neither result is a
forecast improvement or grounds for deployment.

| Frozen output | Training gain vs CV | Held-source gain vs CV | Conditional recording 95% CI |
| --- | ---: | ---: | --- |
| Dense control | +0.187244% | -1.743843% | [-5.899569%, -0.810831%] |
| Expected-cost proposal | +0.195787% | -1.805004% | [-6.079203%, -0.842182%] |
| Expected-cost hard action | 0% | 0% | [0%, 0%] |
| Cost-supervised proposal | +0.184359% | -1.747853% | [-5.838922%, -0.821269%] |
| Cost-supervised hard action | +0.233917% | -1.245558% | [-4.430101%, -0.566922%] |

The primary paired hard-action advantage over dense is +0.498285 percentage
points, conditional CI [+0.242032, +1.438712]. This means **less harm than a
bad neural control**, not positive transfer over CV. The proposal-minus-dense
contrast is -0.004010pp, CI [-0.060179, +0.083615]; no stronger trajectory
decoder is established. All three hard-action seeds lose to CV, ranging from
-1.460677% to -0.958473%.

## Evidence Passport

- `fresh_run`: inference for six frozen endpoints, their exact replays, aggregate
  scoring, 2,000 paired recording-block bootstrap draws and verification.
- `cached_verified`: the completed six-branch training, three dense controls,
  source feature data and all inherited provenance. Dense forecasts are also
  freshly replayed exactly. No new training update was performed in this readout.
- Population: 6,944 complete queries, 181 recording-scoped agents, seven
  recordings, **one** physical site, bookstore. This site is excluded from these
  fits but has been explored previously; it is not independent confirmation.
- Training complement: 15,430 queries, 545 scoped agents, 29 recordings and four
  physical sites. No change to the main task, role assignments or sealed sets.
- Registration `configs/m3w_source_deferral_transfer_v1.json`, committed at
  `92b96b78` before scoring. All six final checkpoints and all three seeds are
  retained. Proposal use requires score > 0; all other outputs use exact CV.
- Seed aggregation averages errors, not trajectories. There is no deployable
  three-seed ensemble hidden in these numbers.
- Bootstrap intervals are conditional within seven recordings of an explored
  site. They are not independent-site or confirmatory confidence statements.

## Safety and Slices

Cost-supervised hard actions intervene on 62.985% of queries. Moving-target
gain is -0.136276%, hard gain -0.036330%, equal-recording gain -3.354742%,
equal-scoped-agent gain -1.460942%, and worst-recording gain -13.996053%.
Thus the negative result is not removed by either recorded weighting sensitivity.
Native annotation-pixel ADE is 0.78358761 versus CV 0.77394765.

The 3,769 zero-target queries incur mean absolute harm 0.01581748 annotation
pixels. CV error is exactly zero on this slice, so percentage easy degradation
is undefined, **not a passed 2% gate**. The hard cutoff is derived exclusively
from training CV error. No physical safety or calibrated-risk claim follows.

The binary future oracle computed from seed-mean hard-action errors gains only
0.138155%; for the raw proposal it is 0.203865%. These are retrospective limits
for the specified fixed error arrays, not executable policies, ensemble scores,
or limits on new architectures. See [full results](results.md).

## Additional Training-Only Diagnostic

The [exact-input audit](training_input_collision_audit.md) groups actual current
mask-model inputs, including the restoration frame, before reading future
labels. Among 15,430 training queries, 153 belong to 32 duplicate-input groups;
only one group, containing 38 rows, has conflicting target paths. No conflicting
group satisfies the tested sufficient condition for zero to be an empirical
ADE minimizer. Large-scale exact input aliasing is not supported as the dominant
explanation. Near-duplicate ambiguity, missing causal cues and limited
learnability remain unresolved. This post-hoc audit is not a population
irreducibility theorem or an additional held evaluation.

The first audit attempt failed while serializing a NumPy count to JSON. Its log
is preserved. A regression test reproduced the issue; explicit boolean
conversion repaired serialization. The successful retry verified the existing
input fingerprints exactly, with no grouping, data or statistical-rule change.
The earlier neural inference and scores were unaffected.

## What Is Next

The intervention rate barely changes from training (61.002%) to held source
(62.985%), but the gain changes sign. In-sample candidate-cost targets may be
optimistic; this is a hypothesis, not an isolated causal explanation.
Before fitting another gate, construct training-side leave-one-site-out
candidate forecasts and gain/harm labels. Keep bookstore outside every inner
fit and reject any parent checkpoint or learned preprocessing that saw its
inner-held site's labels. Report candidate utility before adding a risk head.
This proposed follow-up is `not_run`; no new split or main endpoint is adopted
by this report. A frozen-candidate head comparison may separately diagnose
moving-target optimization, but does not supply novelty by itself.

## Completion Boundary

Nine predictors replay exactly; 15 per-seed output cells and five seed-mean
summaries independently recompute. A repeated evaluation preserves 313 files
with zero updates. Twenty-two focused tests pass; the full legacy suite was
not rerun. See [reproducibility](reproducibility.md), [gates](gates.md), and
[Chinese operations](operation_zh.md).

This source-only experiment is complete and negative. The M3W research goal
remains active and unmet. No new model is promoted. Historical Stage37 results
remain exploratory under the lineage audit, not restored by this experiment.
Offline supplied annotations, stride 12 / +144 raw annotation frames, pixels
and past-normalized coordinates only: no seconds, metric, true-3D, foundation,
human-intention-gold or submission-ready claim. Stage5C and SMC remain off.
