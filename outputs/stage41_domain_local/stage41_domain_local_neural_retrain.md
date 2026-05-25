# Stage41 Domain-Local Neural Endpoint Retrain

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet', 'UCY_expanded']`
- two-domain endpoint gate: `True`

| domain | rows train/val/test | direct all | direct t50 | gated all | gated t50 | gated t100 | gated hard | easy | switch | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `41501/4648/21598` | -0.2358 | -0.7152 | 0.0199 | 0.0036 | 0.0038 | 0.0196 | 0.0000 | 0.0896 | `True` |
| `TrajNet` | `35009/6098/3639` | -0.7538 | -0.5781 | 0.0555 | 0.0628 | 0.0340 | 0.0577 | 0.0000 | 0.2102 | `True` |
| `UCY` | `3490/13254/9540` | -4.8516 | -4.6295 | -0.0043 | -0.0067 | 0.0000 | -0.0002 | 0.1602 | 0.1099 | `False` |
| `UCY_expanded` | `117808/16103/35441` | -0.4303 | -0.1712 | 0.1372 | 0.2625 | 0.0486 | 0.1460 | 0.0000 | 0.4859 | `True` |

This directly trains neural endpoint dynamics per domain using causal seq2seq inputs. Future endpoints are labels/evaluation only. Because this is endpoint-FDE-only, deployment still requires the protected all-agent composite world-state path.

- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
