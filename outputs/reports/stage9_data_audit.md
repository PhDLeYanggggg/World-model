# Stage 9 Data Audit

| item | value |
| --- | --- |
| total per-agent episodes | 320 |
| >=2 agents | 320 |
| >=5 agents | 258 |
| >=10 agents | 191 |
| avg agents | 18.381 |
| max agents | 45 |
| silver episodes | 320 |
| gold episodes | 0 |
| GoalBench official records | 1530 |
| verified t10 episodes | 320 |
| verified t50 episodes | 0 |
| verified t100 episodes | 0 |
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
