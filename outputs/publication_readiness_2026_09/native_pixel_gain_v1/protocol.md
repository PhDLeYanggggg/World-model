# Local past-pixel motion as conditional gain evidence

Use the existing fitting roster and exact observed frames. Compare
three matched conditional gain regressors: no image, background
geometry, and background plus local pixel motion. Keep the original
nomination, quarter amplitude, frozen held-source support model and
support-quartile intervention counts. No coordinate remapping of the
future forecast. No learned image fitting or access to future frames.

Use the existing seven past-only background transforms and quality
features. Separately track five points seeded inside each past head box
to the next already observed image, check forward/backward consistency
and compare supported pixel motion with the annotation-center motion.
Sampling uses previous boxes only; the next box does not fit or gate
the pixel motion. Keep zero-filled unsupported cues with explicit
coverage. These are image-motion measurements, not causal provenance of
the annotation tool, physical occlusion or human-gold tracks.

Fit on eight sources, score the ninth. Three seeds, identical pair
draws and network capacity across arms, 500 updates. Use only pointwise
conditional regression from the prior matched study. Primary contrasts:
background vs no image, and local+background vs background, at exact
support-quartile counts. Include support only, minimum change and OLS4.
Report full-grid paired lower scores with 3,000 source resamples after
seed averaging, conditional gain MAE, ADE/FDE and CV-relative easy harm.
No tuning or unregistered role use; original 2% rule unchanged.
