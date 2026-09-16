# Canonical Causal Recording Rebuild

Fresh raw-position conversion. No legacy teacher or learned cache reused; no training or new performance result.

- Canonical recordings used: 9
- Physical scene groups: ['eth_eth', 'eth_hotel', 'pets09_s2l1', 'ucy_arxiepiskopi', 'ucy_university', 'ucy_zara']
- A collection label such as TrajNet is not an independent dataset domain.
- Named aliases are conservatively co-grouped; only byte-equal aliases are proven numerically identical.
- One incomplete challenge excerpt group is quarantined. No missing future labels are imputed.

| recording | scene group | agents | raw points | 8 observed / 12 future steps | raw10 | raw25 | raw50 | raw100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | eth_eth | 360 | 8908 | 2614 | 0 | 0 | 0 | 0 |
| eth_hotel | eth_hotel | 390 | 6544 | 1197 | 3676 | 0 | 2560 | 1512 |
| ucy_zara01 | ucy_zara | 148 | 5024 | 2234 | 3840 | 0 | 3251 | 2518 |
| ucy_zara02 | ucy_zara | 204 | 9537 | 5741 | 7908 | 0 | 7101 | 6123 |
| ucy_zara03 | ucy_zara | 180 | 3600 | 180 | 2160 | 0 | 1440 | 540 |
| ucy_students01 | ucy_university | 891 | 17820 | 891 | 10692 | 0 | 7128 | 2673 |
| ucy_students03 | ucy_university | 428 | 21846 | 14029 | 18434 | 0 | 16773 | 14784 |
| ucy_arxiepiskopi1 | ucy_arxiepiskopi | 60 | 1200 | 60 | 720 | 0 | 480 | 180 |
| pets09_s2l1 | pets09_s2l1 | 107 | 2140 | 107 | 856 | 0 | 0 | 0 |

## Boundaries

Both protocols are candidate data views, not an approved official benchmark. Observation steps are not seconds. Exact raw horizons require a real label at that frame and a continuous annotation sequence; there is no nearest-future-frame substitution or interpolation.

Past histories, current visible neighbors and causal baseline rollouts are loaded lazily from uncompressed npy arrays. Labels have a separate reader. No central velocity, future endpoint, future availability mask, remaining track length, old selector output, future goal or test normalization enters the input schema.

No scene image, goal map, latent representation or learned dynamics has been added here. Neighbor geometry is observed trajectory context, not evidence of multimodal or physical-world success.

The raw-frame and observed-step views overlap and must never be split independently by rows. Each recording and its aliases must stay together. A strict scene split also keeps Zara01/02/03 together and University recordings together, unlike a merely file-level split.

The local documentation contains unit, FPS and homography claims, but it has not been calibrated against each selected representation here. Dataset-local coordinates and raw-frame/step claims are retained.

The old exposed datasets are development material. No new untouched confirmation set is claimed. Protocol selection, train-only fitting, strong-baseline comparison and independent confirmation remain pending.
