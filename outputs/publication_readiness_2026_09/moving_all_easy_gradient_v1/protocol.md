# Full-Fitting All/Easy Gradient Diagnostic

I test whether the current two risk outputs interfere during optimization before
changing their training rule. The previous membership-BCE/total-cost experiment
did not support a broad gradient-conflict explanation; it tested a different pair.

## Fixed Design

- Use the same three own-B fitting contexts and nine frozen warm, unmasked-MSE,
  and explicit-scale-ablated-MSE checkpoints. Their hashes are verified.
- Preserve the original causal input schema, proposal mask, signed all/easy
  budget-excess supervision, unknown-label bounds and empirical query-draw weights.
- Accumulate both task gradients over the complete fitting data, including empty
  proposal queries in the original denominator. Do not select a favorable batch.
- Compare disposable joint, easy-only and easy-protected full-gradient steps.
  The last removes only the shared all-gradient component opposing the easy
  gradient. Separate output weights are not projected. Saved AdamW state and
  global clipping are retained, rather than treating cosine as an actual update.
- Report before/after task MSE, sign regret and false-safe counts. Restore nothing
  because original model objects and checkpoint files are never updated.

The descriptive trigger for a projected-gradient follow-up requires at least two
of the three original unmasked-MSE contexts to have a shared joint direction
opposing the easy gradient, and at least two to benefit in actual easy MSE from
the protected rather than joint virtual step. This is not a safety or efficacy
gate, nor proof about an entire optimization path or held-scene generalization.
If it fails, do not automatically launch projected-gradient training.

Only fitting B is read for this diagnostic. No new V/H outcome readout, threshold
search, identity certificate, deployment change or independent test. Original
2% harm/easy rules and support requirements remain unchanged. Previous failures
remain failures. The 24-group primary remains not estimable, not passed.
Native obs8/pred12 image-pixel coordinates; no metric/seconds/true-3D/foundation
claim. Stage5C and SMC stay disabled.

Run: `.venv-pytorch/bin/python scripts/diagnose_m3w_moving_all_easy_gradient.py`.
Replay with `--verify`. All checkpoints stay private and outside Git.
