# Protected Forecasting Requires More Than Raw Predictor Accuracy

## Experimental Question

We ask whether a frozen improved neural forecaster remains useful when compared
with an equally protected motion baseline. A predictor-only comparison is
insufficient: a risk controller may suppress the neural model's beneficial
corrections or selectively admit its failures under source shift.

## Design

We use twelve previously opened European source localities, not independent
confirmation data. Three four-locality groups yield six ordered disjoint
producer/controller/readout assignments. Three seeds produce eighteen views
per candidate. Forecasts are frozen; each neural or damped candidate receives
three identically budgeted envelope-bounded cost heads, for positive benefit/harm,
all-cost/harm and positive-easy-cost/harm. Controller preprocessing and labels
use only its fitting localities. Decisions are committed before comparative
outcome readout. Prediction and scoring units are obs8/pred12 at raw stride 12,
with detector-derived silver image coordinates.

Pointwise intervention requires predicted positive gain and both predicted
all/easy harm ratios below 2%. These constraints are model predictions, not a
conformal guarantee. Scene-level controls use fixed hash-selected queries,
an image-proximity objective and verified equal intervention counts. Unmatched
solver outcomes are explicitly separated. Uncertainty resamples twelve locality
means after averaging dependent seeds/producer contexts, with 3,000 bootstrap draws.

## Findings

Protected neural forecasts improve ADE by 0.3342% over CV and improve easy ADE
by 3.9961%; four reference-exact queries are untouched in every dependent view.
However, identically protected damping improves ADE by 0.6994%. The primary neural
advantage is therefore negative: -0.3707% (95% exploratory interval
[-0.6136%, -0.1252%]). The gain from improving the raw forecaster does not transfer
to a competitive protected intervention policy.

Diagnostics reveal both excessive veto and underestimation of selected harm.
The strongest selected-set discrepancy is a predicted ratio of 0.68% versus
12.91% observed. Supported nonadditive edges occur in only 41 neural query-views
out of 6,912 dependent views; matched-count joint selection does not outperform
independent selection. These findings narrow the next method test to reliable
conditional cost learning and supported joint decisions, rather than another
architecture combination or outcome-selected threshold sweep.

## Limitations and Claims Not Made

The source localities have been repeatedly used for development; neither the
bootstrap nor producer exclusion creates independent confirmation. The proxy
does not measure calibrated physical collisions. Labels are silver; time and
coordinate calibration are not established. This is not human-gold, metric,
seconds-level, true3D, foundation or deployment evidence. No novel safety theorem,
latent generative execution or SMC is claimed. A full submission still requires
competitive matched baselines and independent evaluation of the frozen method.
