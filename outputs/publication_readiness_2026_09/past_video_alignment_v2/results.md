# Local Past-Video Alignment Audit

Fresh prefix decoding on fit-only sources. No new model, metric or training modality is admitted.
Requests retain annotation frame indices; source index alignment is inspected, not inferred from playback seconds.

| Source | Encoded rate | Selected agents | Past/current image requests | Unique frames | Full crop support | Old xy bounds | Upstream row/column bounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 25 | 5 | 10 | 8 | 10 | 100.0000% | 100.0000% |
| eth_hotel | 25 | 26 | 52 | 40 | 48 | 82.7781% | 99.8319% |

The source plotting implementation reverses H-inverse output axes. This corrects an availability interpretation,
not any frozen world-coordinate forecast or its scores. Hotel still has out-of-image annotations.
Upstream ETH timing uses15 whereas actual encoded streams report25. Native frame index extraction does not resolve the capture clock.
All requested images were decoded at past/current indices. Decode/index checks do not certify annotation causality,
body detection, per-agent registration, physical scale or sensor-as-of availability.
Raw frames/crops/agent identifiers are local only, excluded from public aggregate reports and Git.
No development/calibration/confirmation labels, Stage5C, SMC, metric/seconds claim or visual-model training.
