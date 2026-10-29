# Literature Position for the Cost/Decision Experiments

Primary-source reading,29October2026. This is related-work material,not an
implemented-method inventory or a claim of a submission-ready contribution.

## Trajectory Modeling Is Not the Novelty Claim

[Trajectron++,ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf)
models heterogeneous agents with graph-structured recurrence,dynamics and map
context. [AgentFormer,ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.html)
jointly models agent/time context through agent-aware attention. These works
already establish that richer social/temporal representations can matter.
Combining a Transformer,neighbors and scene context is not sufficient novelty.

Neither architecture was trained in this four-study control round. No score
comparison to their published tables is valid: data,units,masks,forecast
sampling and observation spacing differ. A matched forecasting implementation
under the same frozen protocol remains a missing paper experiment. Stochastic
sampling in another paper does not authorize Stage5C execution here.

## Multi-Expert Deferral Is Established Prior Work

[Mao,Mohri andZhong,ICML2024](https://proceedings.mlr.press/v235/mao24d.html)
study regression with multiple experts in single-stage and two-stage settings.
The current control adapts their Section4Eq.(3) logistic surrogate to frozen
causal forecasts. It is not a proposed new loss or a reproduction of the
original experiments. Its bounded-loss assumptions and consistency guarantees
have not been verified for our native partial-mask ADE application.

[Cortes etal.,ICML2026](https://proceedings.mlr.press/v306/cortes26b.html)
address expert imbalance with cost-sensitive learning over input/expert pairs
and margin-based deferral. Their MILD method is not implemented in this round.
This paper makes an imbalance-only novelty claim especially weak; any next
imbalance remedy needs an explicit distinction and a matched published control.

## The Open Research Question

The narrower M3W question is whether a causal decision can retain useful neural
forecast substitutions while controlling baseline-relative positive harm on
every required recording,including partial or unknown future labels. Known
mean ADE improvement,net easy ADE preservation and selected positive-harm
ratios are distinct quantities. A safe label-aware oracle demonstrates an
available forecast set,not inference-time identifiability or learned safety.

The current experiment tests continuous moment prediction and recording-level
constraints over a frozen bank. It does not yet solve multi-agent joint scene
selection,prove a population guarantee or provide independent confirmation.
Those gaps cannot be filled by relabeling routing as a world-model advance.

## Risk Calibration Is Not Automatic

[Conformal Risk Control,ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
requires its stated assumptions. Selected harm/reference ratios can increase
when a threshold removes a safe reference row; the loss family is not
automatically monotone. Fitting a threshold on exposed records is not an
independent conformal certificate or an every-record safety guarantee.

Retain the prospective2%contract,undefined empty ratios and utility-only
count matching. Report failed controls and conditional uncertainty. The
scientific next step is a successful,matched,causal decision mechanism and
untouched calibration/confirmation,not a stronger architectural label.
