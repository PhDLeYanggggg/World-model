# Partial Past-Image Support Decision

Purpose: repair the observed all-or-nothing crop failure before any new visual
forecasting experiment. The approved parent remains eight observed/twelve
predicted native annotation steps with its original primary metric. No change
to source roles, labels, holdouts, performance thresholds or observation semantics.

Build a diagnostic-only row-indexed image store from all canonical Zara01/02
source positions, not only rows with complete future targets or complete crops.
Each source row has a fixed 96x96 point-centered crop and 32x32 output. Out-of-frame
pixels remain unobserved: average only available pixels in each nonoverlapping
3x3 block, store the observed count0..9 separately, and set the stored mean to
zero only when a block has no observation. Do not shift crops inward or generate
missing pixels. A genuinely black observed pixel has positive coverage.

All complete eight-step pasts are retained using index lists; no future-survival
condition is applied. Decoding uses source frame = stored frame minus one.
Read-only memory mapping avoids repeated per-window images. No velocities from
the supplied obsmat columns, central differences, endpoint goals or future
target API is used. Source-control provenance is separate from model inputs.

The source annotations remain retrospective/offline. The loader requires an
explicit diagnostic observation mode; the control-as-of-query diagnostic must
reject later-control-dependent queries without silently shrinking the cohort.
Neither mode certifies online identities or sensor availability. Formal training
and evaluation roles remain closed pending the user observation decision.

Verify every real input window, all source row hashes, exact mask semantics,
future-row invariance and completed-cache resume. This is not forecasting
training or a prediction gain. One source row's crop is stored once rather than
eight times. Cache, source images and row-level provenance stay local/ignored.
Source registrations and old results are preserved. Stage5C/SMC remain disabled.
