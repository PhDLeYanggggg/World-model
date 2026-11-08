# Forecasting with causal motion-expert reliability

Prior bounded velocity heads moved very little from OLS4. Instead of
learning another unrestricted correction, mix a fixed family of
causal forecasts: anchored OLS4, last displacement, OLS8, median
velocity, and finite trend. Each expert is clipped relative to OLS4 to
0.35 past RMS step sizes times lead; the learned convex mixture obeys
the same bound. It uses only eight observed positions.

Check whether **recent in-history forecast errors** help determine
which extrapolator to trust: predict the next one or two *observed*
positions from the first five, six, or seven observations. Feed the
scaled OLS4 error and expert differences to the mixture along with
the ego history. The matched history arm zeroes the replay block;
the global mixture zeroes both blocks. All arms use the same model,
initial weights, sampling draws, objective and 800 updates. The loss
is normalized conservative full-grid excess ADE against OLS4 plus
0.25 times available-step excess ADE, fixed before fitting. The
observed term does not change the evaluation or missing-step bound.

Use the existing 69,746-window fitting roster, three fixed seeds and
nine outer source exclusions. Checkpoint every 100 steps, four CPU
threads, no loader subprocesses. Inference is frozen before outcome
comparison. Compare each arm against OLS4, the replay arm against the
matched controls, and report fixed OLS8, median and bounded trend
comparators. Use 3,000 source bootstrap replicates, averaged across
seeds before resampling. Main evidence is positive paired full-grid
lower utility against OLS4; the replay-specific claim additionally
needs positive paired lower utility against the history arm. Report
available-label ADE/FDE, easy/hard diagnostics, and support gaps.
This is fitting-source evidence, not a new deployment selection.

[Scholler et al.](https://arxiv.org/abs/1903.07933) motivates testing
strong motion models. The novelty question here is whether causal
within-history errors add out-of-source information beyond the
same past positions and the same physical expert family, not whether
linear forecasting is novel.

No development, external calibration or confirmation labels. No
change to the original 2% rule or DroneCrowd's sealed 30 groups.
Image pixels and annotation steps only. No deployment, Stage5C or SMC.
