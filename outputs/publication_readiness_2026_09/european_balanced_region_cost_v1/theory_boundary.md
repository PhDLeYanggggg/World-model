# Better Regional Fitting Is Not Decision Calibration

Checked29October2026. The[ICLR2026proceedings metadata](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b8b6df58745740e54f00d84534bd182d-Abstract-Conference.html)
confirms Kiyani,Hassani,Pappas and Roth's paper. The final PDF fetch failed;
OpenReview returned verification pages. Detailed reading here uses the
[authors'v1preprint,sections2-4](https://arxiv.org/html/2510.23471v1),not a
claim that the final conference theorem text was fully reverified.

Their framework constrains forecast residuals with specified test functions
and derives robust decision rules conditional on those calibration guarantees.
Decision-region calibration supports a plug-in decision rule under the stated
setting. The squared-loss discussion uses a linear final predictor. This does
not say that any low-MSE network or any fitting-region regularizer is calibrated.

Our head has nonlinear constrained outputs,and its auxiliary targets concern
known fitting labels within an old teacher's recording regions. Those regions
are not automatically the new model's decision regions. Exact within-region
optimization is therefore not population calibration,a minimax implementation
or an independent every-recording2%guarantee. No such theorem is claimed.

The publication contribution must still demonstrate useful neural intervention
and scene-level decisions beyond strong matched controls,with missing-outcome
bounds and independent evidence. A more accurate regional optimizer alone is
not a new calibration theory or a submission-ready result.
