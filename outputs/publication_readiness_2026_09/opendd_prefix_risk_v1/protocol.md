# Causal Recording-Prefix Risk Allocation

On the existing OpenDD development cohort, freeze the uniform nested
blend and all its forecast weights and alphas. Fit 21 risk heads (seven
outer exclusions, three forecaster seeds). For an outer fold, each of the
six donor sites contributes predictions from the frozen five-site model
that excluded both that donor and the outer site. Use the donor's training
easy cut, excluding the same pair. No changes to the underlying forecasts
or held-site threshold selection. No independent-confirmation access.

The Extra Trees head (32 trees, depth 7, leaf 12, one thread, seed fixed
by producer) estimates five moments: signed gain debited for wholly unknown
outcomes, all and easy positive harm including that debit, and all and
easy reference mass. Train on normalized moments, using site-balanced
weights and only causal eight-step motion, forecast disagreements and
blend size as inputs. Neither labels nor a planned full recording roster
are available to the policy. Forecasts and existing cuts are hash-bound.

For each query in recording time order, choose no edit, one edit, or two
edits (never all visible agents). Select positive predicted gain with
both predicted risk ratios <=2% and positive predicted reference masses.
Query-local starts at zero each query; prefix accumulates only the model's
predicted moments for earlier chosen edits in that same recording, with
no outcome feedback. The prefix-joint arm additionally deducts .25 times
the mean positive reduction in CV pair distance across steps for edges
within 15 publisher units at the final observed frame. This is a geometry
proxy, not a physical safety claim. No risk multipliers are selected using
outcomes. Compare to unchanged CV and the previous uniform blend.

Compare the joint choices to independent utility and additive single-edit
geometry controls on exactly the same number of switches in every query.
They inherit the causal joint count schedule and do not rerun the risk
constraint. These are matched utility controls, not risk-equivalent full
policies. If a required view has zero edits, its matched utility is not
estimable; do not convert it to a zero benefit or safety pass.

Report exact original risk algebra for all 606 recording-seed views,
including undefined selected easy reference and defined violations;
report observed ADE/FDE, full-grid paired lower gain, switch counts and
per-location paired bootstrap intervals (3,000 draws, three-seed average).
Show geometry diagnostics separately. This exposed-source experiment
cannot change the original 2% rule, deployment, Stage5C or SMC.
