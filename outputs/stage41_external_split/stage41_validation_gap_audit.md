# Stage41 Validation Gap Audit

- source: `fresh_run`
- purpose: check whether validation represents held-out t+50 switchability.
- no future endpoint is used as an inference feature; oracle headroom here is a diagnostic label/eval statistic.

| domain | split | rows | t50 rows | t100 rows | oracle all | oracle t50 | candidate6 t50 | hard | easy | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | train | 41501 | 10244 | 8973 | 0.3297 | 0.2868 | -1.0251 | 33026 | 10656 | 16409 |
| ETH_UCY | val | 4648 | 1099 | 717 | 0.2795 | 0.0445 | -1.6628 | 4124 | 1622 | 1775 |
| ETH_UCY | test | 21598 | 5074 | 2614 | 0.4010 | 0.4881 | 0.3271 | 18865 | 1812 | 10862 |
| ETH_UCY | val-test gap |  |  |  |  | 0.4435 | ratio 10.96 |  |  |  |
| TrajNet | train | 35009 | 8448 | 5085 | 0.3620 | 0.3867 | -0.9756 | 25040 | 10917 | 12109 |
| TrajNet | val | 6098 | 1450 | 843 | 0.2479 | 0.0746 | -1.1756 | 4244 | 2495 | 1991 |
| TrajNet | test | 3639 | 831 | 480 | 0.2678 | 0.0963 | -1.2157 | 3116 | 431 | 1854 |
| TrajNet | val-test gap |  |  |  |  | 0.0218 | ratio 1.29 |  |  |  |
| UCY | train | 3490 | 869 | 547 | 0.2936 | 0.0792 | -1.2378 | 2710 | 746 | 1558 |
| UCY | val | 13254 | 3282 | 2009 | 0.4041 | 0.5226 | -0.8793 | 8805 | 4689 | 4087 |
| UCY | test | 9540 | 2340 | 1440 | 0.2933 | 0.0764 | -1.2484 | 7389 | 1979 | 4302 |
| UCY | val-test gap |  |  |  |  | -0.4462 | ratio 0.15 |  |  |  |

## Blockers

- ETH_UCY t50 validation headroom is not representative: val=0.0445, test=0.4881
