# Stage38 External Dataset Audit

- source: `fresh_run`; Stage37 caches are `cached_verified`.

| domain | split | rows | t50 | t100 | k8 | k16 | k32 | k64 | heldout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_UCY | train | 150798 | 37007 | 29328 | 123654 | 86825 | 44059 | 20715 | training_domain |
| TrajNet | train | 8144 | 1936 | 1160 | 4785 | 870 | 0 | 0 | training_domain |
| TrajNet | val | 112746 | 26756 | 16248 | 67879 | 12186 | 0 | 0 | validation_domain |
| UCY | test | 66303 | 16263 | 10008 | 41283 | 7506 | 0 | 0 | heldout_test |

- by domain: `{'UCY': {'rows': 66303, 't50': 16263, 't100': 10008, 'splits': ['test'], 'heldout_test_available': True, 'blocker': None}, 'ETH_UCY': {'rows': 150798, 't50': 37007, 't100': 29328, 'splits': ['train'], 'heldout_test_available': False, 'blocker': 'no held-out test split for frozen Stage37 evaluation; cannot claim deployable generalization'}, 'TrajNet': {'rows': 120890, 't50': 28692, 't100': 17408, 'splits': ['train', 'val'], 'heldout_test_available': False, 'blocker': 'no held-out test split for frozen Stage37 evaluation; cannot claim deployable generalization'}, 'OpenTraj_mixed': {'rows': 337991, 't50': 81962, 't100': 56744, 'splits': ['test', 'train', 'val'], 'heldout_test_available': True, 'blocker': None}}`
