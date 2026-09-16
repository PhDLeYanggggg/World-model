# Fit-Only Context Magnitude

No labels, development rows, model updates or changes to the active protocol.
Counts use complete supervised fit windows and the registered aligned-neighbor rule.

| Recording | Rows | Scale floor | Ego max norm | Neighbor p99 | Neighbor max | Rows neighbor >100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 2614 | 81 | 1.000 | 10577.383 | 16447.400 | 87 |
| eth_hotel | 1197 | 284 | 1.000 | 11813.094 | 14253.628 | 310 |
| ucy_zara01 | 2234 | 0 | 1.000 | 59.141 | 124.316 | 1 |
| ucy_zara02 | 5741 | 0 | 1.000 | 1995.586 | 2424.546 | 2033 |
| ucy_zara03 | 180 | 0 | 0.999 | 532.783 | 911.738 | 7 |

An ego-derived scale can keep ego history near unit magnitude while placing other agents far outside that range.
This is an input-conditioning hypothesis, not proof that neighbors cause the final error or should be removed.
Changing input conditioning must preserve the frozen evaluation scale and baseline comparison, and be separately tested.
No metric/seconds claim, new deployment, independent confirmation, Stage5C or SMC execution.
