# Stage 9 Data Audit

| item | value |
| --- | --- |
| total per-agent episodes | 660 |
| >=2 agents | 660 |
| >=5 agents | 598 |
| >=10 agents | 531 |
| avg agents | 23.965 |
| max agents | 64 |
| silver episodes | 0 |
| gold episodes | 0 |
| GoalBench official records | 1530 |
| verified t10 episodes | 655 |
| verified t50 episodes | 320 |
| verified t100 episodes | 320 |
| stage9 training allowed | True |

Leakage flags:

```json
{
  "test_endpoints_used_for_goals": false,
  "future_endpoint_used_as_input": false,
  "central_velocity_used": false,
  "scene_split_leakage_detected": false
}
```
