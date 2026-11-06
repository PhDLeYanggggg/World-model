# Past-Image Viewpoint Conditioning: A Matched Actor Test

## Material Passport

Previous goal turn is progress: twelve real4000update image-conditioned actors,
one full raw-control refit,two complete exact treatment refits and exact twelve-
bank prediction/scalar replay. The registered whole12motion-minus-quality
component fails:known-1.803852pp,paired lower-10.930870pp,interval
[-17.826116,-5.287792]. All six first-fold donors worsen. All image branches
learn nonzero weights;the worst source has no clipped inputs. These facts do
not establish a unique causal failure mechanism. Old failures remain frozen.

Question:does explicit past-image viewpoint conditioning improve the actual
trajectory actor where an auxiliary image-feature branch failed?

## Fixed Intervention

Keep the pinned EqMotion K1 core,geometry476,nearest eight complete aligned
neighbors,original source sampler/loss factors,AdamW schedule and4000updates.
No new learned modules or parameters. Native observation8/prediction12,stride1.

For seven observed transitions ending at queryq,reuse the previously verified
background affine proxyA_i in native image coordinates. Only when all seven
measurements meet their original fixed support rule,compose each past position
and neighbor into the query image view:T_i=A_q...A_{i+1};T_q=I. Restore each
predicted step withC_t=A_q^t,t=1..12. C_t is an assumed constant-affine future
camera extrapolation computed only from past images,not an observed future pose.
Predictor conditioning is computed from transformed past geometry only. Its
output is restored to native image-relative coordinates before the unchanged
supervised ADE loss. Future targets/masks enter loss/evaluation only.

If any of seven measurements is unsupported,use identity operators for the
entire window. Retain every row and the support flag;identity imputation is not
a measured zero camera motion. The newly trained actor can still differ on
unsupported rows;this rule is not a frozen-model safety fallback or certificate.
No future image,camera pose,endpoint,central velocity or test goal is an input.
No learned data normalization uses donor/test statistics. Query IDs only index
cached past operators;they are not model embeddings or causal features.

The identity control ignores the same operators and is exactly the original
predictor. A full4000update control refit must reproduce original weights,
optimizer,sampler/RNG,draws/factors and common trace before six treatment fits.
One full first-design treatment refit checks complete scientific state. These
are one treatment refit,not six. A real100+100vs200recovery pilot precedes full
fitting. Its200prefix seeds its own unchanged first treatment fit;direct200is
additional compute. Total real-data updates32,200if uninterrupted.

Also freeze a simple camera-compensated causal CV forecast. It uses the same
past composition and extrapolation without a neural network. This is a new
strong-control diagnostic,not an outcome-selected primary or a new risk floor.
Report the original six causal baselines as diagnostic reference controls.

## Data And Runtime

Same125540exposed producer-A windows,14sources,34recordings,1613queries and six
source-excluding fitting designs. Reuse original native geometry,labels and
source exclusions;no own-B,C/H or reserved outcome access. Source separation
within each fit is not independent confirmation. One forecaster seed17 is an
exploratory component test,not the intended three-seed formal main result.

Preparation opens past metadata and past-pair receipts only. Generic later
loading opens/hashes the full packed training bank,including labels,for cache
integrity. Only past geometry/operators reach inference;forecast freeze occurs
before diagnostic scoring,not before all target-file byte access.

Nativearm64CPU4/inter-op1/workers0,no resource probing or multiprocessing.
Atomic checkpoints every200updates,loss/heartbeat every50,same-definition
resume,12hphase boundary,2GiBreserve and512MiBnew-private artifact cap. Latest
scientific state is also saved on interruption when storage permits. Existing
data and checkpoints are not deleted. Healthy slow training is not reduced.
Pilot measures memory/time. CREATE is checked read-only;observation timeout
does not imply any remote job is stopped or the queue is empty.

## Frozen Readout And Decision

Freeze all six actor and six camera-CV donor banks before new outcome scoring.
Primary:new camera actor minus cached identity actor over available12ADE.
Same source/query-balanced prefix1/3/6/12statistics,partial/unknown rows and
paired completion lower. Same3000source-bootstrap draws,seed39107.
Progression requires positive whole12lower and CI lower,at least8/14positive
known-source contrasts,and both producer-fold known means positive. Report all
sources,recordings,easy/hard diagnostics and unsupported-camera strata without
deleting a slice or changing the primary after readout. Camera-CV minus raw CV
and actor minus camera-CV are secondary controls. Complete prediction/scalar
replay and full first-design refit are required.

This screen only informs whether to investigate a separately registered
controller. It is not the full original per-recording2%all/easy positive-harm,
signed easy-net,reference/support/NA contract. No identity certificate,epsilon
denominator,threshold search,new controller or deployment. No verified metric,
seconds,homography,true3D,foundation,human-gold or physical-safety claim.
Stage5C execution and SMC remain off.
