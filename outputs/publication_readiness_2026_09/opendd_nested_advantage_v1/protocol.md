# Nested low-capacity relative-advantage selection

Status: **draft protocol, not registered, nothing fitted or scored.** The
runner, `registration.json` and tests do not exist yet. The draft becomes
binding only once the author approves the open decisions in the last section.
Registration then commits it, hash-bound, before any new fit. This is exposed
OpenDD development evidence, not external confirmation.

## Question

`opendd_onpolicy_donors_v1` fitted its donor task (selected-pass predicted
44.73% vs observed 45.18%, ECE 1.80%) but failed to transfer to held locations
(67.20% vs 20.79%, ECE 52.87%). Its paired optional selector scored 52/606
full-grid passes against 55 for keeping the incumbent. Three explanations
remain confounded:

1. **Target mismatch.** Selectors trained on the original-dual label but were
   judged by the stricter full-grid certificate.
2. **Absolute vs relative target.** The model predicts each option's own
   pass. The decision only needs the change relative to keeping the incumbent.
3. **Capacity.** A 48/32 GELU MLP may overfit location-specific structure.

This experiment asks whether either of the following transfers better than
the current head, in ranking as well as calibration, on donor-held
validation locations that no upstream model has seen:

- a lower-capacity model;
- a relative keep-versus-switch advantage target trained on the full-grid
  label.

Only after that inner selection is fixed do the outer choices get frozen and
scored. Feature set, slate construction, budget and evaluation roster stay
unchanged.

## Arms

All arms use the same causal exchange features as v1 (`exp.design(...,
'aligned')`) and the same donor slates. They differ only in target, loss and
capacity.

| Arm | Model | Target and loss | Decision |
| --- | --- | --- | --- |
| `ref48` | v1 `ListRanker` 48/32 GELU, keep row via `with_incumbent` | original-dual, BCE + pairwise vs keep (identical to v1 `pair`) | argmax over keep + exchanges, ties keep |
| `abs_lin` | linear, keep row via `with_incumbent` | full-grid pass, BCE over keep + exchanges, no pairwise term | argmax over keep + exchanges, ties keep |
| `abs_mlp16` | one hidden layer, 16 units, GELU, keep row | as `abs_lin` | as `abs_lin` |
| `adv_lin` | linear, exchange rows only, 3 logits | full-grid advantage `a_j = y_j - y_keep` in {-1, 0, +1}, softmax cross-entropy | switch score `p(+1) - p(-1)`; take argmax exchange only if its score > 0, else keep |
| `adv_mlp16` | one hidden layer, 16 units, GELU, 3 logits | as `adv_lin` | as `adv_lin` |

Fixed for every arm:

- **Optimizer:** AdamW, lr 1e-3, PyTorch default weight decay.
- **Training:** 400 updates; 32 slates per batch; same `batcher` (uniform
  donor site, then slate); seeds 17/29/43, tied to producers. Each slate gets
  equal weight, and padded options are masked.
- **Standardization:** `abs_*` and `adv_*` standardize features with mean and
  standard deviation (floor 1e-6) from that cell's training donor option rows
  only. The keep row's zeroed option block is not counted. `ref48` keeps v1's
  raw inputs so it replicates v1.
- **Switching rules:** the advantage switch margin is fixed at 0, with no
  threshold, margin, temperature or confidence search. There are no class
  weights. Advantage arms do not see the keep row, so the incumbent's own
  absolute pass is never estimated.

Forced variants (best exchange, never keep) are reported as diagnostics only.

## Labels

The full-grid label is v1's held-location evaluation certificate, applied to
donor slates:

- the original dual pass;
- full-grid risk upper bound ≤ 2%;
- guaranteed easy support;
- easy excess ≤ 0.

It is computed with the existing `certificate`
(`scripts/audit_m3w_opendd_full_grid_exchange.py`, `LIMIT = .02`). The easy
cut comes from the producer's permitted fit locations, as in v1. Labels are
read only after logits, incumbent, exchange pairs and features are fixed (v1
`donor` order). Undefined full-grid support counts as not passing, as in the
certificate. Its frequency is reported per role. The original-dual label is
kept alongside for `ref48` and for diagnostics.

## Nested roles

There are seven exposed OpenDD development locations. Notation:

- outer held location H;
- inner validation location V ≠ H;
- donor location D ∉ {H, V};
- teacher target T.

An upstream model may never have seen any location whose labels later judge
it.

| Role | Nominee ranker excludes | Count head excludes | Source |
| --- | --- | --- | --- |
| Inner training donor slate on D for (H, V) | {H, V, D} | {H, V, D} | ranker: reuse v1 triple `rank_*`; count head: **new** |
| Teacher for the new count head, on T | {H, V, D, T} | — | **new** quadruple ranker |
| Inner validation slate on V for H | {H, V} | {H, V} | reuse v1 count head + archived double ranker (same producers as v1 donor `{H}_{V}`) |
| Outer training donor slate on D for H | {H, D} | {H, D} | reuse, same as v1 |
| Outer held views on H | frozen v1 evaluation roster | — | unchanged |

Why the new count heads are needed: v1 count head `{H, D}` was taught on V.
Using its donor slates to fit an inner selector and then validating on V would
leak V through the incumbent allocation. Old double-excluded models must not
stand in for these roles.

**New upstream fits:**

- 35 quadruple exclusions × 3 seeds = 105 rankers (`SwapValue`, 700 updates,
  fitted on 3 locations);
- 35 triple exclusions × 3 seeds = 105 count heads (`CountHead`, 500 updates,
  fitted on 4 locations, each target taught by its quadruple ranker);
- 420 new teacher caches.

All cutoffs come from the respective model's own fit locations.

**Reuse with hash check:**

- the 105 v1 triple rankers, now as inner donor nominee producers;
- the 63 v1 count heads;
- the archived double rankers.

The v1 triple rankers served as teacher producers in v1. Their class and step
count match the donor nominee role (`SwapValue(False)`, 700 updates). The
registration records this role change explicitly.

**Caches:**

- 315 inner donor caches (21 excluded pairs × 5 donors × 3 seeds);
- 126 relabelled v1-producer caches. These serve as both inner validation
  slates and outer training donors.

The relabelled caches must reproduce v1 donor features, anchors and original
labels bit-for-bit. Only the full-grid columns are new. A mismatch stops the
run.

## Inner selection: per outer location, H-blind

For each ordered (H, V) and seed, fit all five arms on the five donors
D ∉ {H, V}. That is 42 × 3 × 5 = 630 fits and 252,000 updates. Score the
optional choices on V's validation slates.

**Selection metric** `M_H(arm)`: the full-grid pass rate of the arm's
optional choice minus keep, in percentage points. Average it over seeds within
V, then weight the six validation locations equally.

**Choice:** select `argmax M_H`. Ties go to lower capacity, in this order:
`adv_lin`, `abs_lin`, `adv_mlp16`, `abs_mlp16`, `ref48`. No other quantity
enters the choice.

**Reported per H, but not used for selection:**

- rescue/destroy counts;
- switch rate;
- within-slate ranking AUC on heterogeneous slates;
- ten-bin ECE and Brier score;
  - absolute arms: calibration on the selected option's pass;
  - advantage arms: calibration on its predicted vs observed advantage;
- the same metrics on the donors' own training slates, giving the train/held
  gap.

The validation slates are v1-style donor slates: all recordings with a legal
exchange and nonzero reference mass, with exchanges subsampled to 32. They are
not the outer 101-view eligibility mask. This population difference is
recorded rather than corrected.

Each H's selection is written to `inner_selection.json` and committed before
any outer fit.

## Outer fit and freeze

For each H and seed, fit all five arms on the six outer donors D ≠ H, using
the relabelled v1-producer caches. That is 7 × 3 × 5 = 105 fits and 42,000
updates. The **primary policy** for H is the arm chosen by inner selection.
The other four are frozen alongside as secondary arms and cannot replace it
afterwards.

All 21 location/seed choice sets are frozen and committed before scoring.

**Total configured updates:**

| Component | Updates |
| --- | ---: |
| New rankers: 105 × 700 | 73,500 |
| New count heads: 105 × 500 | 52,500 |
| Inner fits | 252,000 |
| Outer fits | 42,000 |
| **Total** | **420,000** |

## Fixed readout

The evaluation constraints are unchanged:

- the 101 eligible views out of 606 recording-seed views;
- 780 forecast edits;
- count, movement and token caps;
- the empty-support treatment;
- the original 2% rule.

Statistics: 3,000 paired-location bootstrap draws, seed 817. Seeds are
averaged within each location before resampling. Aggregation uses
`total_fields` so that zero-valued keys are retained.

**Primary:** the inner-selected policy, optional choices, full-grid passes.
The repair is supported only if all three lower 95% bounds are strictly
positive:

1. against keeping the incumbent;
2. against uniform exchange on the policy's own switch mask;
3. against v1's paired optional choices.

One favourable contrast is not success.

**Also reported:**

- all five arms, forced and optional;
- original-dual and material full-grid passes;
- rescue/destruction;
- risk-undefined and risk-violation counts;
- per-location and per-seed tables;
- the same-mask hindsight oracle, labelled diagnostic;
- held calibration and ranking metrics, mirroring the inner ones.

The outer result is reported even if the inner selection already shows that
no arm beats keep.

## Not changed or authorized

The following stay unchanged:

- outer thresholds and keep margins;
- external WEPDTOF/Songdo/Athens labels and release policies;
- DroneCrowd held confirmation;
- the author Word file, v219 PDFs and replay;
- disclosures.

Stage5C and SMC stay disabled. Recording-level pooled history features and the
batch allocator are not a streaming-deployment claim. A pass on this exposed
development set would justify a separately registered confirmation, not a
deployment.

## Runtime

- **Environment:** native arm64 `.venv-pytorch`, 4 CPU threads, 1 interop
  thread, 0 workers.
- **Checkpoints:** model, optimizer and RNG every 100 updates; process lock,
  heartbeat and a per-invocation log.
- **Pilot:** one triple-excluded count cell with its four quadruple rankers,
  then one inner (H, V, seed) cell. The measured runtime and RSS decide local
  vs CREATE.
- **Expected scale:** v1 ran 121,800 updates locally in about 5 minutes of
  fitting, so local is expected.
- **Limits:** keep a 10 GiB disk reserve. Stop at a 12-hour checkpoint
  boundary. Resume by re-running the same phase.
- **Phases:** `register`, `producers`, `inner`, `select`, `fit`, `score`.
  Nothing is scored from an incomplete matrix.

## Prior work boundary

These earlier experiments already covered related ground:

- `opendd_candidate_advantage_v1` (query-level gain/harm);
- `opendd_slate_rank_v1` (pairwise ranking);
- `opendd_nested_spline_choice_v2` (nested calibration exclusion);
- `opendd_credit_value_v2` (credit/timing advantage).

The new elements here are limited to:

- nested exclusion that also removes the inner validation location from
  upstream producers;
- a full-grid relative-advantage target for the whole allocation keep/exchange
  decision.

Neither a three-class advantage head nor a capacity ablation is claimed as a
method contribution.

## Open decisions before registration

1. **Arm set.** Five arms (2 targets × 2 capacities + `ref48`), or drop `ref48`
   from the inner selection and keep it only as an outer reference?
2. **Inner metric.** Select on utility only (pass rate minus keep), with
   calibration reported, or also require an inner ECE ceiling?
3. **Inner failure.** If every arm has `M_H ≤ 0` for most H, do we still fit
   and score the outer primary (current draft: yes, to avoid selective
   reporting), or stop after the inner readout?
4. **Advantage label for undefined support.** Count it as not passing (current
   draft, matching the certificate), or exclude those options from advantage
   training?
