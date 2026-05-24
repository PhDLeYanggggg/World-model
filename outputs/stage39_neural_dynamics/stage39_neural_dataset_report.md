# Stage39 Neural Dataset Report

- source: `fresh_run`
- reports: `{'train': {'split': 'train', 'source': 'cached_verified', 'rows': 24000, 'path': 'data/stage39_neural_dynamics/neural_dataset_train.npz'}, 'val': {'split': 'val', 'source': 'cached_verified', 'rows': 8000, 'path': 'data/stage39_neural_dynamics/neural_dataset_val.npz'}, 'test': {'split': 'test', 'source': 'cached_verified', 'rows': 16000, 'path': 'data/stage39_neural_dynamics/neural_dataset_test.npz'}}`
- no leakage: `{'inputs': 'past-only history, neighbor history proxies, goal prototypes, Stage37 selected baseline rollout, domain/horizon labels', 'future_endpoint_input': False, 'future_labels_only_for_loss_eval': True, 'central_velocity': False, 'test_endpoint_goals': False, 'num_workers': 0}`
