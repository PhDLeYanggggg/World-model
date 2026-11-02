# Post-Readout Selected Cost Attribution

This diagnosis was added after the completed calibration readout. It does not
choose a multiplier, fit a model, alter decisions or open confirmation.
Native predicted cost moments are compared on exactly the same known-label
actual-switch rows as the original risk metric. The lambda-scaled optimizer
coefficient view is not treated as a predicted cost distribution.

|Lambda|Policy|Median predicted/true H|Median predicted/true EH|Median predicted/true R|Median predicted/true ER|All violations (harm-dominant)|Easy violations (harm-dominant)|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|independent|0.201091|2.96084e-05|1.40482|0.0115603|830 (639)|876 (854)|
|1|joint_matched|0.195606|3.10265e-05|1.04854|0.0130079|882 (739)|902 (885)|
|2|independent|0.195145|1.86082e-05|1.76362|0.0113938|764 (522)|859 (834)|
|2|joint_matched|0.164468|1.93198e-05|1.32685|0.0118659|847 (669)|894 (867)|
|4|independent|0.173249|8.29526e-06|2.26386|0.00897943|661 (360)|837 (799)|
|4|joint_matched|0.154432|7.79982e-06|1.73807|0.00956466|772 (557)|867 (833)|
|8|independent|0.165894|4.62484e-06|2.88248|0.0069959|533 (268)|825 (769)|
|8|joint_matched|0.148251|4.55553e-06|2.33906|0.00755131|682 (393)|845 (791)|
|16|independent|0.151446|1.86578e-06|3.69989|0.00622257|429 (189)|777 (715)|
|16|joint_matched|0.145251|1.68965e-06|3.03753|0.00632859|552 (299)|789 (735)|
|32|independent|0.118186|5.90323e-07|4.87523|0.005406|369 (132)|706 (646)|
|32|joint_matched|0.119905|6.0948e-07|4.15722|0.00492244|433 (192)|728 (673)|
|64|independent|0.0904872|1.44758e-07|6.38138|0.00339223|332 (86)|625 (578)|
|64|joint_matched|0.096417|1.3591e-07|5.55528|0.00312274|365 (104)|650 (598)|
|128|independent|0.0716727|2.85367e-08|8.77182|0.00150666|281 (47)|480 (443)|
|128|joint_matched|0.0744023|2.35077e-08|7.71631|0.00140969|295 (53)|511 (475)|
|256|independent|0.0492973|3.5929e-09|11.7729|0.000637184|242 (27)|388 (362)|
|256|joint_matched|0.0500927|3.76526e-09|11.134|0.000655099|255 (34)|408 (382)|
|512|independent|0.0384279|3.87931e-10|15.9116|0.000250142|222 (22)|283 (273)|
|512|joint_matched|0.0379812|4.52955e-10|15.5285|0.000244331|224 (23)|297 (285)|
|1024|independent|0.0269574|4.49796e-11|23.7381|0.000133293|168 (12)|197 (185)|
|1024|joint_matched|0.0260302|4.56511e-11|23.0908|0.000133293|176 (13)|211 (198)|
|2048|independent|0.0175251|1.87778e-12|25.7245|4.86143e-05|116 (3)|129 (120)|
|2048|joint_matched|0.01651|2.3134e-12|25.2522|4.71356e-05|118 (3)|130 (121)|
|4096|independent|0.0108455|2.92595e-14|27.3579|1.09703e-05|86 (1)|90 (85)|
|4096|joint_matched|0.0103443|3.63045e-14|26.8282|8.52548e-06|89 (1)|92 (87)|

The exact identity is observed excess minus predicted excess =
harm underprediction + 0.02 times reference overprediction, separately for
H/R and EH/ER. Negative components alleviate the excess. A dominant component
is an accounting diagnosis, not proof of the causal training failure.
Medians exclude undefined ratios and are explicitly conditional; the JSON
retains finite counts, unknown selected rows and native cost masses. Views
repeat nine pipelines and are not independent observations. Missing outcomes
are not imputed. Available-label cost is not full12 ADE. No physical safety,
metric/seconds, true3D, foundation, Stage5C or SMC claim is made.
