# Direct Kinematic Forecast: Matched Peer Information Test

**Question.** Can adaptive smoothing of observed velocities yield a more
useful trajectory than fixed history smoothing, and do peers improve it?
The previous warm PCT penalty reduced unknown-row disagreement but hurt
paired conservative utility. This test replaces free-form trajectory output,
not the evaluator or its support requirements.

**Method.** Each step integrates a softmax-weighted mixture of seven past
velocity increments multiplied by a learned gain in (0, 2). This bounds step
speed by twice the maximum observed speed. It is a direct motion forecast,
not a baseline correction or an identity fallback certificate. The peer arm
adds a masked mean of encoded causal relative distances and velocities. The
matched self arm replaces this context with zeros; parameter initialization,
fitting rows, draws and step budget stay fixed. No visibility or target
signals enter the predictor.

**Fit.** The verified fold-2 roster has 69,746 fitting rows, nine nonempty
sources and one declared zero-support source. Six fresh native Torch fits:
two information arms by actual seeds 17/29/43, 2,400 steps per arm and seed,
128 samples per update, previous fitting-only source factors, masked ADE.
Fixed final step, no A selection. Prefix checkpoint at 20, resume and atomic
checkpoints each 100 steps thereafter. CPU4/interop1/workers0.

**Readout.** Freeze all six predictions across all 125,540 A rows before
opening A outcomes. A is already exposed development, not confirmation.
Primary paired diagnostic: completion-lower gain against fitting-selected
OLS4. Compare both arms against CV and EqMotion, and peer against self.
Retain direct paired bounds, 3,000 paired source resamples after seed
aggregation, per-record ADE/FDE availability, easy/hard and original
selected-reference risk. Independently replay the old oracle ceiling, then
add the new paths without exporting label-aware choices. A ceiling is not
policy performance. The original 2% and reference/support rules, all 102
recording views and the 8/12 image-coordinate protocol stay fixed. No
external scoring, deployment, residual head, Stage5C or SMC.
