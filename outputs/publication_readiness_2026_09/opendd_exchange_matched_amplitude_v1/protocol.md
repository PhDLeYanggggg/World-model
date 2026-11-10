# OpenDD: timing at matched cumulative deviation

The first timing test retained peak displacement but did not keep
the sum of per-step deviations. Make a stricter comparison: transfer
norm mass between the first and second halves of the receiving
trajectory. Freeze direction at each step, peak steps, the final
position, and total norm mass. Distribute the transfer proportionally
to removable mass and receiving headroom with a fixed sine-squared
taper, taking half the lesser capacity. No target coordinates enter
the transfer or the prefix-OLS phase choice.

Use the same 7,198 legal exchanges, 606 recording-seed views, 780 edits,
101 matched ranker exchanges, utilities, stress deletion test, and
original 2% rule. Report exact nested capacity and paired location
intervals for the unchanged, front-loaded, back-loaded, uniform,
and prefix-OLS shapes. This is development only; no new thresholds
or external outcomes.
