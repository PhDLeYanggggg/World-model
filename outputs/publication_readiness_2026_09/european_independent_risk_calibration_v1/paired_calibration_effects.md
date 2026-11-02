# Paired Calibration Policy Effects

All fixed-grid contrasts are descriptive, not another selector or parameter search.
Intervals resample 12 locality groups after pooling native reference mass within
each locality. Calibration selection is not undone by these intervals.
The unknown-row term is the actual policy-pair path disagreement, not the
difference of two separate floor-relative lower bounds. Missing steps inside
partially observed trajectories remain unaddressed.

|Lambda|Contrast|Known pp [CI95]|Paired lower pp [CI95]|
|---:|---|---:|---:|
|1|joint_matched_minus_independent|0.513428 [0.270137, 0.801437]|0.069670 [-0.094556, 0.262792]|
|1|joint_free_minus_independent|0.845941 [0.540747, 1.205455]|0.287871 [0.079581, 0.521491]|
|1|frame_uniform_minus_independent|-0.386716 [-0.637662, -0.171938]|-0.833033 [-1.173513, -0.527642]|
|2|joint_matched_minus_independent|0.408639 [0.194987, 0.664631]|0.151554 [0.005592, 0.343793]|
|2|joint_free_minus_independent|0.640786 [0.386256, 0.937153]|0.317751 [0.136068, 0.534475]|
|2|frame_uniform_minus_independent|-0.194450 [-0.333397, -0.079402]|-0.482764 [-0.693816, -0.310531]|
|4|joint_matched_minus_independent|0.272025 [0.132681, 0.428124]|0.140078 [0.033377, 0.268737]|
|4|joint_free_minus_independent|0.424408 [0.262273, 0.610949]|0.237933 [0.099577, 0.401546]|
|4|frame_uniform_minus_independent|-0.095807 [-0.173811, -0.025608]|-0.283829 [-0.428692, -0.169453]|
|8|joint_matched_minus_independent|0.122822 [0.071056, 0.178073]|0.050961 [0.002520, 0.110137]|
|8|joint_free_minus_independent|0.231121 [0.154221, 0.307763]|0.115726 [0.032706, 0.198610]|
|8|frame_uniform_minus_independent|-0.059613 [-0.108575, -0.018142]|-0.188648 [-0.304003, -0.106320]|
|16|joint_matched_minus_independent|0.051776 [0.029917, 0.073789]|0.005321 [-0.024076, 0.035746]|
|16|joint_free_minus_independent|0.096409 [0.054398, 0.142474]|0.036813 [-0.004273, 0.079199]|
|16|frame_uniform_minus_independent|-0.038876 [-0.080715, -0.001116]|-0.116994 [-0.180863, -0.066650]|
|32|joint_matched_minus_independent|0.014464 [0.005840, 0.021910]|0.002216 [-0.011937, 0.014524]|
|32|joint_free_minus_independent|0.040409 [0.022912, 0.058181]|0.004606 [-0.040106, 0.035700]|
|32|frame_uniform_minus_independent|-0.039835 [-0.078064, -0.006885]|-0.104650 [-0.159911, -0.052399]|
|64|joint_matched_minus_independent|0.018797 [-0.000135, 0.038242]|0.009751 [-0.003448, 0.023513]|
|64|joint_free_minus_independent|0.035456 [0.007827, 0.065098]|0.020675 [-0.005870, 0.047590]|
|64|frame_uniform_minus_independent|-0.015760 [-0.032314, 0.004297]|-0.042244 [-0.068156, -0.018835]|
|128|joint_matched_minus_independent|0.011411 [0.001894, 0.023363]|0.006525 [-0.004977, 0.018812]|
|128|joint_free_minus_independent|0.019660 [0.006735, 0.034563]|0.010098 [-0.004522, 0.024896]|
|128|frame_uniform_minus_independent|-0.007237 [-0.019588, 0.008477]|-0.023661 [-0.042651, -0.006703]|
|256|joint_matched_minus_independent|0.003489 [-0.000039, 0.008423]|0.002784 [-0.000323, 0.006929]|
|256|joint_free_minus_independent|0.010073 [0.002197, 0.019382]|0.009108 [0.001489, 0.018153]|
|256|frame_uniform_minus_independent|-0.003428 [-0.013133, 0.009521]|-0.016299 [-0.030359, -0.003944]|
|512|joint_matched_minus_independent|0.000051 [-0.003276, 0.003881]|-0.000151 [-0.003513, 0.003683]|
|512|joint_free_minus_independent|0.000791 [-0.003303, 0.004526]|0.000563 [-0.003545, 0.004370]|
|512|frame_uniform_minus_independent|-0.006025 [-0.012105, -0.001504]|-0.013394 [-0.025617, -0.003312]|
|1024|joint_matched_minus_independent|0.000129 [-0.000847, 0.001243]|0.000126 [-0.000847, 0.001240]|
|1024|joint_free_minus_independent|0.002181 [0.000146, 0.004832]|0.002147 [0.000096, 0.004801]|
|1024|frame_uniform_minus_independent|-0.002859 [-0.005324, -0.000969]|-0.005512 [-0.011691, -0.001205]|
|2048|joint_matched_minus_independent|0.000004 [-0.000047, 0.000065]|0.000002 [-0.000050, 0.000064]|
|2048|joint_free_minus_independent|0.001200 [-0.000096, 0.003484]|0.001198 [-0.000099, 0.003484]|
|2048|frame_uniform_minus_independent|-0.001009 [-0.002144, -0.000132]|-0.003718 [-0.007876, -0.000464]|
|4096|joint_matched_minus_independent|0.000223 [-0.000027, 0.000705]|0.000223 [-0.000029, 0.000704]|
|4096|joint_free_minus_independent|0.000230 [-0.000026, 0.000711]|0.000229 [-0.000027, 0.000710]|
|4096|frame_uniform_minus_independent|-0.000360 [-0.001407, 0.000510]|-0.002748 [-0.006651, 0.000099]|
