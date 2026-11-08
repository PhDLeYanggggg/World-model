# Direct Velocity Forecasts With Past Road Context

Development experiment on the same seven-site OpenDD roster, past-only
3,395 queries and 10,475 pedestrian targets. The earlier ranking runs
changed only selection. Here we train new forecast trajectories and
test whether traffic context helps even before selection. No threshold
search, original-risk revision or DroneCrowd access.

Use 8 observed and 12 predicted positions at the original diagnostic
stride. No global coordinates, site ID, publisher velocities, maps,
vehicle future, endpoint information or label-validity inputs. In the
heading-aligned target frame, encode its complete observed history and
current causal speed. For context, use up to two nearest other pedestrian
histories in the query and up to two nearest motor road users (Car, Van,
Truck, Motorcycle) within 15 publisher coordinate units at the final
observed sample. Neighbors are selected using past/current positions and
IDs only. Missing past steps are masked; no interpolation or future fill.
Coordinates and time remain specific to the released OpenDD export.

Use a matched scalar-width MLP for three input arms: self-only; self plus
pedestrian neighbors; self plus pedestrians and motor traffic. Ablated
channels are zeroed, not removed. The output is three bounded 2-D velocity
knots interpolated over 12 future steps and integrated from the last
observed position. This is a direct deterministic candidate, not residual
policy training or latent generation. Every arm receives the same draws,
initialization, optimizer and loss. Fit 600 AdamW updates (.001, batch
128, site then target uniformly), three seeds 17/29/43 and seven
leave-location-out folds. Train on available future steps with normalized
smooth Euclidean error plus 0.25 positive CV-relative excess; no missing
future labels enter optimization as positions. Save every 100 steps.

Report direct forecast and fixed 25% interpolation toward the new path
against CV and OLS4. Primary context contrast is motor+pedestrian vs
pedestrian-only with identical capacity and aligned targets. Score
observed ADE/FDE, conservative paired gain on all scheduled steps,
descriptive easy ADE, worst site and 3,000 paired equal-site resamples
after averaging seeds. The offline one-edit oracle compares the original
motion menu with the menu plus each new forecast at the exact same
query count; only the oracle reads outcomes for decisions. Also report
the same paired effects on rows with motor context, as a descriptive
past-defined stratum (omit sites with no observed labels in that stratum).
No deployment, external confirmation, Stage5C or SMC.
